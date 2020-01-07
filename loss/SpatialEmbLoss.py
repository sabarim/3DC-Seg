"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import torch
import torch.nn as nn
import numpy as np
import pdb
from loss.LossUtils import parse_embedding_output, precision_tensor_to_matrix, mahalanobis_distance
from loss.LovaszLoss import lovasz_hinge


class SpatioTemporalEmbLoss(nn.Module):

    def __init__(self, embedding_size=3, to_center=True, n_sigma=1, foreground_weight=1):
        super().__init__()

        # print('Created spatial emb loss function with: to_center: {}, n_sigma: {}, foreground_weight: {}'.format(
        #     to_center, n_sigma, foreground_weight))

        self.to_center = to_center and embedding_size == 3
        self.n_sigma = n_sigma
        self.foreground_weight = foreground_weight
        assert embedding_size >= 3
        self.embedding_size = embedding_size

        # coordinate map
        x = torch.linspace(0, 4.16, 2000).view(
            1, 1, 1, -1).expand(1, 32, 800, 2000)
        y = torch.linspace(0, 1.6, 800).view(
            1, 1, -1, 1).expand(1, 32, 800, 2000)
        t = torch.linspace(0, 0.1, 32).view(
            1, -1, 1, 1).expand(1, 32, 800, 2000)
        xyzm = torch.cat((t, y, x), 0)

        self.register_buffer("xyzm", xyzm)

    def forward(self, prediction, instances, labels, w_inst=1, w_var=10, w_seed=1, iou=False, iou_meter=None):

        batch_size, tw, height, width = prediction.size(
            0), prediction.size(2), prediction.size(-2), prediction.size(-1)

        xyzm_s = self.xyzm[:, 0:tw, 0:height, 0:width].contiguous() # 3 x t x h x w
        if self.embedding_size > 3:
            xyzm_s = torch.cat((xyzm_s, torch.zeros(self.embedding_size - 3, tw, height, width)), dim=0)

        loss = 0

        for b in range(0, batch_size):

            emb = torch.tanh(prediction[b, 0:self.embedding_size]) # e x t x h x w
            spatial_emb = emb[0:3] + xyzm_s  # e x t x h x w
            sigma = prediction[b, self.embedding_size:self.embedding_size+self.n_sigma]  # n_sigma x t x h x w
            seed_map = torch.sigmoid(prediction[b, self.embedding_size+self.n_sigma:self.embedding_size+self.n_sigma + 1])  # 1 x t x h x w

            # loss accumulators
            var_loss = []
            instance_loss = []
            seed_loss = []
            obj_count = 0

            instance = instances[b]  #  1 x t x h x w
            # label = labels[b].unsqueeze(0)  # 1 x h x w

            instance_ids = instance.unique()
            instance_ids = instance_ids[instance_ids != 0]

            # regress bg to zero
            bg_mask = instance == 0
            if bg_mask.sum() > 0:
                seed_loss += [torch.sum(
                    torch.pow(seed_map[bg_mask] - 0, 2))]

            for id in instance_ids:

                in_mask = instance.eq(id)   # 1 x t x h x w

                # calculate center of attraction
                if self.to_center:
                    xyz_in = xyzm_s[in_mask.expand_as(xyzm_s)].view(3, -1)
                    center = xyz_in.mean(1).view(3, 1, 1, 1)  # 3 x 1 x 1
                else:
                    center = spatial_emb[in_mask.expand_as(spatial_emb)].view(
                        self.embedding_size, -1).mean(1).view(self.embedding_size, 1, 1, 1)  # e x 1 x 1 x 1

                # calculate sigma
                sigma_in = sigma[in_mask.expand_as(
                    sigma)].view(self.n_sigma, -1)

                s = sigma_in.mean(1).view(
                    self.n_sigma, 1, 1)   # n_sigma x 1 x 1

                # calculate var loss before exp
                # NOTE: detach the loss before accumulation to avoid memory leak due to pytorch graph creation
                var_loss = var_loss + \
                    [torch.mean(
                        torch.pow(sigma_in - s.detach(), 2))]

                s = torch.exp(s*10)

                # calculate gaussian
                dist = torch.exp(-1*torch.sum(
                    torch.pow(spatial_emb - center, 2)*s.unsqueeze(1), 0, keepdim=True))

                # apply lovasz-hinge loss
                instance_loss = instance_loss + \
                    [lovasz_hinge(dist*2-1, in_mask.float())]

                # seed loss
                seed_loss += [self.foreground_weight * torch.sum(
                    torch.pow(seed_map[in_mask] - dist[in_mask].detach(), 2))]

                # calculate instance iou
                if iou:
                    iou_meter.update(calculate_iou(dist > 0.5, in_mask))

                obj_count += 1

            instance_loss = torch.stack(instance_loss).sum(dim=0)
            var_loss = torch.stack(var_loss).sum(dim=0)
            if obj_count > 0:
                instance_loss /= obj_count
                var_loss /= obj_count

            seed_loss = torch.stack(seed_loss).sum(dim=0) / (height * width)

            loss += w_inst * instance_loss + w_var * var_loss + \
                    w_seed * seed_loss

        loss = loss / (b+1)

        return loss


class SpatioTemporalLossWithFloatingCentre(nn.Module):
    def __init__(self, embedding_size=3, to_center=True, n_sigma=1, foreground_weight=1):
        super().__init__()

        # print('Created spatial emb loss function with: to_center: {}, n_sigma: {}, foreground_weight: {}'.format(
        #     to_center, n_sigma, foreground_weight))

        self.to_center = to_center and embedding_size == 3
        self.n_sigma = n_sigma
        self.foreground_weight = foreground_weight
        assert embedding_size >= 3
        self.embedding_size = embedding_size

        # coordinate map
        x = torch.linspace(0, 4.16, 2000).view(
            1, 1, 1, -1).expand(1, 32, 800, 2000)
        y = torch.linspace(0, 1.6, 800).view(
            1, 1, -1, 1).expand(1, 32, 800, 2000)
        t = torch.linspace(0, 0.1, 32).view(
            1, -1, 1, 1).expand(1, 32, 800, 2000)
        xyzm = torch.cat((t, y, x), 0)

        self.register_buffer("xyzm", xyzm)

    def forward(self, prediction, instances, labels, w_inst=1, w_var=10, w_seed=1, iou=False, iou_meter=None):

        batch_size, tw, height, width = prediction.size(
            0), prediction.size(2), prediction.size(-2), prediction.size(-1)

        xyzm_s = self.xyzm[:, 0:tw, 0:height, 0:width].contiguous()  # 3 x t x h x w
        if self.embedding_size > 3:
            xyzm_s = torch.cat((xyzm_s, torch.zeros(self.embedding_size - 3, tw, height, width)), dim=0)

        loss = 0

        for b in range(0, batch_size):

            emb = torch.tanh(prediction[b, 0:self.embedding_size])  # e x t x h x w
            spatial_emb = emb[0:3] + xyzm_s  # e x t x h x w
            sigma = prediction[b, self.embedding_size:self.embedding_size + self.n_sigma]  # n_sigma x t x h x w
            seed_map = torch.sigmoid(prediction[b,
                                     self.embedding_size + self.n_sigma:self.embedding_size + self.n_sigma + 1])  # 1 x t x h x w

            # loss accumulators
            var_loss = 0
            instance_loss = 0
            seed_loss = 0
            obj_count = 0

            instance = instances[b]  # 1 x t x h x w
            # label = labels[b].unsqueeze(0)  # 1 x h x w

            instance_ids = instance.unique()
            instance_ids = instance_ids[instance_ids != 0]

            # regress bg to zero
            bg_mask = instance == 0
            if bg_mask.sum() > 0:
                seed_loss += torch.sum(
                    torch.pow(seed_map[bg_mask] - 0, 2))

            for id in instance_ids:

                in_mask = instance.eq(id)  # 1 x t x h x w

                # calculate center of attraction
                centres = [spatial_emb[:, t][in_mask[:, t].expand_as(spatial_emb[:, t])].view(
                    self.embedding_size, -1).mean(1).view(self.embedding_size, 1, 1) for t in range(tw)]  # [e x 1 x 1,...]

                # calculate sigma
                sigma_ins = [sigma[:, t][in_mask[:, t].expand_as(
                    sigma[:, t])].view(self.n_sigma, -1) for t in range(tw)]

                s = [sigma_in.mean(1).view(
                    self.n_sigma, 1, 1) for sigma_in in sigma_ins]  # n_sigma x 1 x 1

                # calculate var loss before exp
                var_losses = torch.stack([torch.mean(
                               torch.pow(sigma_ins[t] - s[t].detach(), 2)) for t in range(len(s))])
                #FIXME: var_losses for frames without any instances are ignored
                var_losses = var_losses[torch.isnan(var_losses) != True]
                var_loss = var_loss + \
                           torch.mean(var_losses)
                
                s = torch.stack(s).permute(1,0,2,3)
                s = torch.exp(s* 10) # t
                centres = torch.stack(centres)
                ignore_mask = torch.isnan(centres) != True
                centre = centres[ignore_mask].mean(dim=0)

                # calculate gaussian
                ignore_mask = torch.any(ignore_mask, dim=1).squeeze()
                dist = torch.exp(-1 * torch.sum(
                    torch.pow(spatial_emb - centre.unsqueeze(-1), 2) * s, 0, keepdim=True))
                # apply lovasz-hinge loss
                instance_loss = instance_loss + \
                        lovasz_hinge(dist[:, ignore_mask] * 2 - 1, in_mask[:, ignore_mask].float())
        
                # seed loss
                seed_loss += self.foreground_weight * torch.sum(
                        torch.pow(seed_map[in_mask] - dist[in_mask].detach(), 2))

                # calculate instance iou
                if iou:
                    iou_meter.update(calculate_iou(dist[:, ignore_mask] > 0.5, in_mask[:, ignore_mask]))

                obj_count += 1

            if obj_count > 0:
                instance_loss /= obj_count
                var_loss /= obj_count

            seed_loss = seed_loss / (height * width)

            loss += w_inst * instance_loss + w_var * var_loss + w_seed * seed_loss

        loss = loss / (b + 1)

        return loss


class CovarianceLoss(nn.Module):
    def __init__(self, embedding_size=3, to_center=True, n_sigma=1, foreground_weight=1):
        super().__init__()

        # print('Created spatial emb loss function with: to_center: {}, n_sigma: {}, foreground_weight: {}'.format(
        #     to_center, n_sigma, foreground_weight))

        self.to_center = to_center and embedding_size == 3
        self.n_sigma = n_sigma
        self.foreground_weight = foreground_weight
        assert embedding_size >= 3
        self.embedding_size = embedding_size

        # coordinate map
        x = torch.linspace(0, 4.16, 2000).view(
            1, 1, 1, -1).expand(1, 32, 800, 2000)
        y = torch.linspace(0, 1.6, 800).view(
            1, 1, -1, 1).expand(1, 32, 800, 2000)
        t = torch.linspace(0, 0.1, 32).view(
            1, -1, 1, 1).expand(1, 32, 800, 2000)
        xyzm = torch.cat((t, y, x), 0)

        self.register_buffer("xyzm", xyzm)

    def forward(self, prediction, instances, labels, w_inst=1, w_var=10, w_seed=1, iou=False, iou_meter=None):

        batch_size, tw, height, width = prediction.size(
            0), prediction.size(2), prediction.size(-2), prediction.size(-1)

        xyzm_s = self.xyzm[:, 0:tw, 0:height, 0:width].contiguous()  # 3 x t x h x w
        if self.embedding_size > 3:
            xyzm_s = torch.cat((xyzm_s, torch.zeros(self.embedding_size - 3, tw, height, width)), dim=0)

        loss = 0

        for b in range(0, batch_size):

            emb = torch.tanh(prediction[b, 0:self.embedding_size])  # e x t x h x w
            spatial_emb = emb[0:3] + xyzm_s  # e x t x h x w
            sigma = prediction[b, self.embedding_size:self.embedding_size + self.n_sigma]  # n_sigma x t x h x w
            seed_map = torch.sigmoid(prediction[b,
                                     self.embedding_size + self.n_sigma:self.embedding_size + self.n_sigma + 1])  # 1 x t x h x w

            # loss accumulators
            var_loss = []
            instance_loss = []
            seed_loss = []
            obj_count = 0

            instance = instances[b]  # 1 x t x h x w
            # label = labels[b].unsqueeze(0)  # 1 x h x w

            instance_ids = instance.unique()
            instance_ids = instance_ids[instance_ids != 0]

            # regress bg to zero
            bg_mask = instance == 0
            if bg_mask.sum() > 0:
                seed_loss += torch.sum(
                    torch.pow(seed_map[bg_mask] - 0, 2))

            for id in instance_ids:

                in_mask = instance.eq(id)  # 1 x t x h x w

                # calculate center of attraction
                if False:
                    xyz_in = xyzm_s[in_mask.expand_as(xyzm_s)].view(3, -1)
                    center = xyz_in.mean(1).view(3, 1, 1, 1)  # 3 x 1 x 1
                else:
                    center = spatial_emb[in_mask.expand_as(spatial_emb)].view(
                        self.embedding_size, -1).mean(1).view(1, self.embedding_size)  # 1 x e

                # calculate sigma
                sigma_in = sigma[in_mask.expand_as(
                    sigma)].view(self.n_sigma, -1)

                precision_vals = parse_embedding_output(sigma_in.permute(1,0), self.embedding_size)
                precision_mat = precision_tensor_to_matrix(precision_vals, self.embedding_size)
                assert precision_mat.shape == (self.embedding_size, self.embedding_size)
                # print("precision_mat: {}".format(precision_mat))

                s = precision_vals.mean(0).view(
                    1, self.n_sigma)  # 1 x n_sigma

                # calculate var loss before exp
                var_loss = var_loss + \
                           [torch.mean(
                               torch.pow(precision_vals - s.detach(), 2))]

                # s = torch.exp(s * 10)

                # calculate gaussian
                dist = torch.exp(-1 * mahalanobis_distance(spatial_emb.reshape(self.embedding_size, -1).permute(1,0),
                                                           center=center, precision_mat=precision_mat,
                                         return_squared_distance=True))
                dist = dist.reshape(in_mask.shape[1:]).unsqueeze(0)

                # apply lovasz-hinge loss
                instance_loss = instance_loss + \
                                [lovasz_hinge(dist * 2 - 1, in_mask.float())]

                # seed loss
                seed_loss += [self.foreground_weight * torch.sum(
                    torch.pow(seed_map[in_mask] - dist[in_mask].detach(), 2))]

                # calculate instance iou
                if iou:
                    iou_meter.update(calculate_iou(dist > 0.5, in_mask))

                obj_count += 1

            instance_loss = torch.stack(instance_loss).sum(dim=0)
            var_loss = torch.stack(var_loss).sum(dim=0)
            if obj_count > 0:
                instance_loss /= obj_count
                var_loss /= obj_count

            seed_loss = torch.stack(seed_loss).sum(dim=0)
            seed_loss = seed_loss / (height * width)

            loss += w_inst * instance_loss + w_var * var_loss + w_seed * seed_loss

        loss = loss / (b + 1)

        return loss


class SpatialEmbLoss(nn.Module):
    def __init__(self, to_center=True, n_sigma=1, foreground_weight=1, ):
        super().__init__()

        # print('Created spatial emb loss function with: to_center: {}, n_sigma: {}, foreground_weight: {}'.format(
        #     to_center, n_sigma, foreground_weight))

        self.to_center = to_center
        self.n_sigma = n_sigma
        self.foreground_weight = foreground_weight

        # coordinate map
        xm = torch.linspace(0, 2, 2048).view(
            1, 1, -1).expand(1, 1024, 2048)
        ym = torch.linspace(0, 1, 1024).view(
            1, -1, 1).expand(1, 1024, 2048)
        xym = torch.cat((xm, ym), 0)

        self.register_buffer("xym", xym)

    def forward(self, predictions, instances, labels, w_inst=1, w_var=10, w_seed=1, iou=False, iou_meter=None):

        batch_size, time, height, width = predictions.size(
            0), predictions.size(-3), predictions.size(-2), predictions.size(-1)

        xym_s = self.xym[:, 0:height, 0:width].contiguous()  # 2 x h x w

        loss = 0

        for b in range(0, batch_size):
            for t in range(time):
                prediction = predictions[:, :, t]
                spatial_emb = torch.tanh(prediction[b, 0:2]) + xym_s  # 2 x h x w
                sigma = prediction[b, 2:2 + self.n_sigma]  # n_sigma x h x w
                seed_map = torch.sigmoid(
                    prediction[b, 2 + self.n_sigma:2 + self.n_sigma + 1])  # 1 x h x w

                # loss accumulators
                var_loss = 0
                instance_loss = 0
                seed_loss = 0
                obj_count = 0

                instance = instances[b, :, t]  # 1 x h x w
                # label = labels[b].unsqueeze(0)  # 1 x h x w

                instance_ids = instance.unique()
                instance_ids = instance_ids[instance_ids != 0]

                # regress bg to zero
                bg_mask = instance == 0
                if bg_mask.sum() > 0:
                    seed_loss += torch.sum(
                        torch.pow(seed_map[bg_mask] - 0, 2))

                for id in instance_ids:

                    in_mask = instance.eq(id)  # 1 x h x w

                    # calculate center of attraction
                    if self.to_center:
                        xy_in = xym_s[in_mask.expand_as(xym_s)].view(2, -1)
                        center = xy_in.mean(1).view(2, 1, 1)  # 2 x 1 x 1
                    else:
                        center = spatial_emb[in_mask.expand_as(spatial_emb)].view(
                            2, -1).mean(1).view(2, 1, 1)  # 2 x 1 x 1

                    # calculate sigma
                    sigma_in = sigma[in_mask.expand_as(
                        sigma)].view(self.n_sigma, -1)

                    s = sigma_in.mean(1).view(
                        self.n_sigma, 1, 1)  # n_sigma x 1 x 1

                    # calculate var loss before exp
                    var_loss = var_loss + \
                               torch.mean(
                                   torch.pow(sigma_in - s.detach(), 2))

                    s = torch.exp(s * 10)

                    # calculate gaussian
                    dist = torch.exp(-1 * torch.sum(
                        torch.pow(spatial_emb - center, 2) * s, 0, keepdim=True))

                    # apply lovasz-hinge loss
                    instance_loss = instance_loss + \
                                    lovasz_hinge(dist * 2 - 1, in_mask.float())

                    # seed loss
                    seed_loss += self.foreground_weight * torch.sum(
                        torch.pow(seed_map[in_mask] - dist[in_mask].detach(), 2))

                    # calculate instance iou
                    if iou:
                        iou_meter.update(calculate_iou(dist > 0.5, in_mask))

                obj_count += 1

            if obj_count > 0:
                instance_loss /= obj_count
                var_loss /= obj_count

            seed_loss = seed_loss / (height * width)

            loss += w_inst * instance_loss + w_var * var_loss + w_seed * seed_loss

        loss = loss / b

        return loss + prediction.sum() * 0


def calculate_iou(pred, label):
    intersection = ((label == 1) & (pred == 1)).sum()
    union = ((label == 1) | (pred == 1)).sum()
    if not union or union ==0:
        return 0
    else:
        iou = np.nan_to_num(intersection.item() / union.item())
        return iou
