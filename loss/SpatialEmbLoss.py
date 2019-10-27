"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import torch
import torch.nn as nn

from loss.LovaszLoss import lovasz_hinge


class SpatioTemporalEmbLoss(nn.Module):

    def __init__(self, to_center=True, n_sigma=1, foreground_weight=1):
        super().__init__()

        # print('Created spatial emb loss function with: to_center: {}, n_sigma: {}, foreground_weight: {}'.format(
        #     to_center, n_sigma, foreground_weight))

        self.to_center = to_center
        self.n_sigma = n_sigma
        self.foreground_weight = foreground_weight

        # coordinate map
        xm = torch.linspace(0, 1, 480).view(
            1, 1, 1,-1).expand(1, 8, 480, 480)
        ym = torch.linspace(0, 1, 480).view(
            1, 1, -1, 1).expand(1, 8, 480, 480)
        zm = torch.linspace(0, 0.15, 8).view(
            1, -1, 1,1).expand(1, 8, 480, 480)
        xyzm = torch.cat((xm, ym, zm), 0)

        self.register_buffer("xyzm", xyzm)

    def forward(self, prediction, instances, labels, w_inst=1, w_var=10, w_seed=1, iou=False, iou_meter=None):

        batch_size, tw, height, width = prediction.size(
            0), prediction.size(2), prediction.size(-2), prediction.size(-1)

        xyzm_s = self.xyzm[:, 0:tw, 0:height, 0:width].contiguous().cuda() # 2 x h x w

        loss = 0

        for b in range(0, batch_size):

            spatial_emb = torch.tanh(prediction[b, 0:3]) + xyzm_s  # 3 x t x h x w
            sigma = prediction[b, 3:3+self.n_sigma]  # n_sigma x t x h x w
            seed_map = torch.sigmoid(prediction[b, 3+self.n_sigma:3+self.n_sigma + 1])  # 1 x t x h x w

            # loss accumulators
            var_loss = 0
            instance_loss = 0
            seed_loss = 0
            obj_count = 0

            instance = instances[b]  #  1 x t x h x w
            # label = labels[b].unsqueeze(0)  # 1 x h x w

            instance_ids = instance.unique()
            instance_ids = instance_ids[instance_ids != 0]

            # regress bg to zero
            bg_mask = instance == 0
            if bg_mask.sum() > 0:
                seed_loss += torch.sum(
                    torch.pow(seed_map[bg_mask] - 0, 2))

            for id in instance_ids:

                in_mask = instance.eq(id)   # 1 x t x h x w

                # calculate center of attraction
                if self.to_center:
                    xyz_in = xyzm_s[in_mask.expand_as(xyzm_s)].view(3, -1)
                    center = xyz_in.mean(1).view(3, 1, 1, 1)  # 3 x 1 x 1
                else:
                    center = spatial_emb[in_mask.expand_as(spatial_emb)].view(
                        3, -1).mean(1).view(3, 1, 1, 1)  # 3 x 1 x 1

                # calculate sigma
                sigma_in = sigma[in_mask.expand_as(
                    sigma)].view(self.n_sigma, -1)

                s = sigma_in.mean(1).view(
                    self.n_sigma, 1, 1)   # n_sigma x 1 x 1

                # calculate var loss before exp
                var_loss = var_loss + \
                    torch.mean(
                        torch.pow(sigma_in - s.detach(), 2))

                s = torch.exp(s*10)

                # calculate gaussian
                dist = torch.exp(-1*torch.sum(
                    torch.pow(spatial_emb - center, 2)*s.unsqueeze(1), 0, keepdim=True))

                # apply lovasz-hinge loss
                instance_loss = instance_loss + \
                    lovasz_hinge(dist*2-1, in_mask.float())

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

        loss = loss / (b+1)

        return loss + prediction.sum()*0


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
                spatial_emb = torch.tanh(prediction[b, 0:2]) + xym_s.cuda()  # 2 x h x w
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
                        center = xy_in.mean(1).view(2, 1, 1).cuda()  # 2 x 1 x 1
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
                        torch.pow(spatial_emb - center, 2) * s.cuda(), 0, keepdim=True))

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

        loss = loss / (b + 1)

        return loss + prediction.sum() * 0

def calculate_iou(pred, label):
    intersection = ((label == 1) & (pred == 1)).sum()
    union = ((label == 1) | (pred == 1)).sum()
    if not union:
        return 0
    else:
        iou = intersection.item() / union.item()
        return iou
