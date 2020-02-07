"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import collections
import os
import threading

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.nn import PairwiseDistance

from inference_handlers.infer_utils.Visualisation import visualize_embeddings
from loss.LossUtils import parse_embedding_output, precision_tensor_to_matrix, mahalanobis_distance
from loss.SpatialEmbLoss import calculate_iou
from util import get_best_overlap


PROBABILITY_THRESHOLD = 0.7


class AverageMeter(object):
  def __init__(self, num_classes=1):
    self.num_classes = num_classes
    self.reset()
    self.lock = threading.Lock()

  def reset(self):
    self.sum = [0] * self.num_classes
    self.count = [0] * self.num_classes
    self.avg_per_class = [0] * self.num_classes
    self.avg = 0

  def update(self, val, cl=0):
    with self.lock:
      self.sum[cl] += val
      self.count[cl] += 1
      self.avg_per_class = [
        x / y if x > 0 else 0 for x, y in zip(self.sum, self.count)]
      self.avg = sum(self.avg_per_class) / len(self.avg_per_class)


class Visualizer:
  def __init__(self, keys):
    self.wins = {k: None for k in keys}

  def display(self, image, key):

    n_images = len(image) if isinstance(image, (list, tuple)) else 1

    if self.wins[key] is None:
      self.wins[key] = plt.subplots(ncols=n_images)

    fig, ax = self.wins[key]
    n_axes = len(ax) if isinstance(ax, collections.Iterable) else 1

    assert n_images == n_axes

    if n_images == 1:
      ax.cla()
      ax.set_axis_off()
      ax.imshow(self.prepare_img(image))
    else:
      for i in range(n_images):
        ax[i].cla()
        ax[i].set_axis_off()
        ax[i].imshow(self.prepare_img(image[i]))

    plt.draw()
    self.mypause(0.001)

  def savePlt(self, image, key, path):

    n_images = len(image) if isinstance(image, (list, tuple)) else 1

    if self.wins[key] is None:
      self.wins[key] = plt.subplots(ncols=n_images)

    fig, ax = self.wins[key]
    n_axes = len(ax) if isinstance(ax, collections.Iterable) else 1

    assert n_images == n_axes

    if n_images == 1:
      ax.cla()
      ax.set_axis_off()
      ax.imshow(self.prepare_img(image))
    else:
      for i in range(n_images):
        ax[i].cla()
        ax[i].set_axis_off()
        ax[i].imshow(self.prepare_img(image[i]))
    # plt.draw()
    plt.savefig(path)

  @staticmethod
  def prepare_img(image):
    if isinstance(image, Image.Image):
      return image

    if isinstance(image, torch.Tensor):
      image.squeeze_()
      image = image.numpy()

    if isinstance(image, np.ndarray):
      if image.ndim == 3 and image.shape[0] in {1, 3}:
        image = image.transpose(1, 2, 0)
      return image

  @staticmethod
  def mypause(interval):
    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
      figManager = matplotlib._pylab_helpers.Gcf.get_active()
      if figManager is not None:
        canvas = figManager.canvas
        if canvas.figure.stale:
          canvas.draw()
        canvas.start_event_loop(interval)
        return


class Cluster:
  def __init__(self, ):
    # coordinate map
    x = torch.linspace(0, 4.16, 2000).view(
      1, 1, 1, -1).expand(1, 32, 800, 2000)
    y = torch.linspace(0, 1.6, 800).view(
      1, 1, -1, 1).expand(1, 32, 800, 2000)
    t = torch.linspace(0, 0.1, 32).view(
      1, -1, 1, 1).expand(1, 32, 800, 2000)
    xyzm = torch.cat((t, y, x), 0)

    # coordinate map
    # x = torch.linspace(0, 2, 960).view(
    #   1, 1, 1, -1).expand(1, 32, 512, 960)
    # y = torch.linspace(0, 1, 512).view(
    #   1, 1, -1, 1).expand(1, 32, 512, 960)
    # t = torch.linspace(0, 0.1, 32).view(
    #   1, -1, 1, 1).expand(1, 32, 512, 960)
    # xyzm = torch.cat((t, y, x), 0)

    self.xyzm = xyzm

  def cluster_with_gt(self, prediction, instance, n_sigma=1, ):

    height, width = prediction.size(1), prediction.size(2)

    xym_s = self.xyzm[:, 0:height, 0:width]  # 2 x h x w

    spatial_emb = torch.tanh(prediction[0:2]) + xym_s  # 2 x h x w
    sigma = prediction[2:2 + n_sigma]  # n_sigma x h x w

    instance_map = torch.zeros(height, width).byte().cuda()

    unique_instances = instance.unique()
    unique_instances = unique_instances[unique_instances != 0]

    for id in unique_instances:
      mask = instance.eq(id).view(1, height, width)

      center = spatial_emb[mask.expand_as(spatial_emb)].view(
        2, -1).mean(1).view(2, 1, 1)  # 2 x 1 x 1

      s = sigma[mask.expand_as(sigma)].view(n_sigma, -1).mean(1).view(n_sigma, 1, 1)
      s = torch.exp(s * 10)  # n_sigma x 1 x 1

      dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb - center, 2) * s, 0))

      proposal = (dist > 0.5)
      instance_map[proposal] = id

    return instance_map

  def cluster(self, prediction, n_sigma=1, threshold=0.5, iou_meter=None, in_mask=None, visualise_clusters=False,
              floating = False):
    if floating:
      return self.cluster_floating(prediction, n_sigma, threshold, iou_meter, in_mask)
    else:
      return self.cluster_with_mahalanobis(prediction, n_sigma, threshold, iou_meter, in_mask) \
        if prediction.shape[0] != (n_sigma*2+1) else \
        self.cluster_squared(prediction, n_sigma, threshold, iou_meter, in_mask, visualise_clusters=visualise_clusters)

  def cluster_squared(self, prediction, n_sigma=1, threshold=0.5, iou_meter=None, in_mask=None, visualise_clusters=False):

    time, height, width = prediction.size(-3), prediction.size(-2), prediction.size(-1)
    embedding_dim = prediction.shape[0] - (n_sigma + 1)

    xyzm_s = self.xyzm[:, 0:time, 0:height, 0:width].contiguous()  # 3 x t x h x w
    if embedding_dim > 3:
      xyzm_s = torch.cat((xyzm_s, torch.zeros(embedding_dim - 3, time, height, width,
                                              device=prediction.device)), dim=0)
    spatial_emb = torch.tanh(prediction[0:embedding_dim]) + xyzm_s  # 3 x t x h x w
    sigma = prediction[embedding_dim:embedding_dim + n_sigma]  # n_sigma x t x h x w
    # seed_map = torch.sigmoid(prediction[3 + n_sigma:3 + n_sigma + 1])  # 1 x t x h x w
    seed_map = prediction[embedding_dim + n_sigma:embedding_dim + n_sigma + 1]  # 1 x t x h x w

    instance_map = torch.zeros(time, height, width).byte()
    instances = []


    count = 1
    # mask = seed_map > 0.5
    mask = seed_map.bool()
    vis=None
    if mask.sum() > 128*time:

      spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(embedding_dim, -1)
      sigma_masked = sigma[mask.expand_as(sigma)].view(n_sigma, -1)
      seed_map_masked = seed_map[mask].view(1, -1)

      labels = torch.cat(((in_mask.sum(dim=0) ==0)[None].byte(), in_mask))
      labels = torch.argmax(labels, dim=0)[None]
      labels = labels[mask.expand_as(labels)]

      if visualise_clusters:
        vis = visualize_embeddings(spatial_emb_masked.permute(1,0), labels.reshape(-1),
                                   torch.exp(sigma_masked*10).permute(1,0), True)

      unclustered = torch.ones(mask.sum()).byte()
      instance_map_masked = torch.zeros(mask.sum()).byte()

      # track used masks for computing iou
      used_ids = {}
      while (unclustered.sum() > 128):
        seed = (seed_map_masked * unclustered.float()).argmax().item()
        seed_score = (seed_map_masked * unclustered.float()).max().item()
        if seed_score < threshold:
          break
        center = spatial_emb_masked[:, seed:seed + 1]
        unclustered[seed] = 0
        s = torch.exp(sigma_masked[:, seed:seed + 1] * 10)
        dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb_masked -
                                                  center, 2) * s, 0, keepdim=True))

        proposal = (dist > PROBABILITY_THRESHOLD).squeeze()

        if proposal.sum() > 128:
          if unclustered[proposal].sum().float() / proposal.sum().float() > 0.5:
            instance_map_masked[proposal.squeeze()] = count
            instance_mask = torch.zeros(time, height, width).int()
            instance_mask[mask.squeeze().cpu()] = proposal.int().cpu()
            instances.append(
              {'mask': instance_mask.squeeze(), 'score': seed_score, 'centre': center})
            count += 1
            # calculate instance iou
            if iou_meter is not None and in_mask.shape[1] > 0:
              iou, id = get_best_overlap(instance_mask.numpy(),
                               in_mask.data.cpu().numpy())
              if id == -1:
                iou_meter.update(0)
              elif id not in used_ids.keys():
                used_ids[id] = calculate_iou(instance_mask.squeeze(), in_mask[id].squeeze())
              elif iou > used_ids[id]:
                used_ids[id] = calculate_iou(instance_mask.squeeze(), in_mask[id].squeeze())
            elif in_mask.shape[1] == 0:
              iou_meter.update(0)

        unclustered[proposal] = 0

      instance_map[mask.squeeze().cpu()] = instance_map_masked.cpu()
      if in_mask.shape[1] == 0 and len(instances):
        iou_meter.update(1)
      else:
        iou_meter.update(np.nan_to_num(np.mean(list(used_ids.values()))))

    return instance_map, instances, vis

  def cluster_floating(self, prediction, n_sigma=1, threshold=0.5, iou_meter = None, in_mask = None):

    time, height, width = prediction.size(-3), prediction.size(-2), prediction.size(-1)
    xyzm_s = self.xyzm[:, 0:time, 0:height, 0:width]
    embedding_dim = prediction.shape[0] - (n_sigma + 1)

    spatial_emb = torch.tanh(prediction[0:3]) + xyzm_s  # 3 x t x h x w
    sigma = prediction[3:3 + n_sigma]  # n_sigma x t x h x w
    # seed_map = torch.sigmoid(prediction[3 + n_sigma:3 + n_sigma + 1])  # 1 x t x h x w
    seed_map = prediction[3 + n_sigma:3 + n_sigma + 1]  # 1 x t x h x w

    instance_map = torch.zeros(time, height, width).byte()
    local_instance_map = torch.ones(time, height, width).byte()
    instances = []

    count = 1
    # mask = seed_map > 0.5
    mask = seed_map.bool()

    if mask.sum() > 128:
      spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(3, -1)
      sigma_masked = sigma[mask.expand_as(sigma)].view(n_sigma, -1)
      seed_map_masked = seed_map[mask].view(1, -1)

      unclustered = torch.ones(mask.sum()).byte()
      instance_map_masked = torch.zeros(mask.sum()).byte()

      # track used masks for computing iou
      used_ids = {}
      while (unclustered.sum() > 128*time):
        seed = (seed_map_masked * unclustered.float()).argmax().item()
        seed_score = (seed_map_masked * unclustered.float()).max().item()
        if seed_score < threshold:
          break

        # get the tube centre
        center = spatial_emb_masked.view(3, -1)[:, seed:seed + 1] # e x 1
        # masked embeddings
        valid_emb=spatial_emb * mask.expand_as(spatial_emb)
        valid_sigma = (sigma * mask.expand_as(sigma)).view(3, time, -1)
        valid_emb = valid_emb.view(3, time, -1)
        local_centres = (valid_emb - center[:, None,:]).view(3,time,-1)
        centre_indices = torch.argmin(local_centres, dim=-1)
        local_centres = torch.stack([
          torch.stack([valid_emb[d, idx][centre_indices[d, idx]] for idx in range(time)])
          for d in range(embedding_dim)])  # e x t x 1

        unclustered[seed] = 0

        local_sigma = torch.stack([
          torch.stack([valid_sigma[d, idx][centre_indices[d, idx]] for idx in range(time)])
          for d in range(embedding_dim)])
        s = torch.exp(local_sigma * 10)
        dist = torch.exp(-1 * torch.sum(torch.pow(valid_emb -
                                                  center[:, None], 2) * s[..., None], 0, keepdim=True))

        proposal = (dist > PROBABILITY_THRESHOLD).squeeze()
        proposal = proposal[mask.reshape(time, width*height)]
        if proposal.sum() > 128*time:
          if unclustered[proposal].sum().float() / proposal.sum().float() > 0.5:
            instance_map_masked[proposal.squeeze()] = count
            instance_mask = torch.zeros(time, height, width).int()
            instance_mask[mask.squeeze().cpu()] = proposal.int().cpu()
            instances.append(
              {'mask': instance_mask.squeeze(), 'score': seed_score, 'centre': center})
            count += 1
            # calculate instance iou
            if iou_meter is not None and in_mask.shape[1] > 0:
              iou, id = get_best_overlap(instance_mask.numpy(),
                               in_mask.squeeze().data.cpu().numpy())
              if id not in used_ids.keys():
                used_ids[id] = calculate_iou(instance_mask.squeeze(), in_mask[id])
              elif iou > used_ids[id]:
                used_ids[id] = calculate_iou(instance_mask.squeeze(), in_mask[id])
              elif -1:
                iou_meter.update(0)
            elif in_mask.shape[1] == 0:
              iou_meter.update(0)
        unclustered[proposal] = 0

      instance_map[mask.squeeze().cpu()] = instance_map_masked.cpu()
      if in_mask.shape[1] == 0 and len(instances == 0):
        iou_meter.update(1)
      else:
        iou_meter.update(np.nan_to_num(np.mean(list(used_ids.values()))))

    return instance_map, instances, None

  def cluster_with_mahalanobis(self, prediction, n_sigma=1, threshold=0.5, iou_meter=None, in_mask=None):

    time, height, width = prediction.size(-3), prediction.size(-2), prediction.size(-1)
    xyzm_s = self.xyzm[:, 0:time, 0:height, 0:width]
    embedding_size = prediction.shape[0] - (n_sigma + 1)

    spatial_emb = torch.tanh(prediction[0:3]).cuda() + xyzm_s.cuda()  # 3 x t x h x w
    sigma = prediction[3:3 + n_sigma].cuda()  # n_sigma x t x h x w
    # seed_map = torch.sigmoid(prediction[3 + n_sigma:3 + n_sigma + 1])  # 1 x t x h x w
    seed_map = prediction[3 + n_sigma:3 + n_sigma + 1].cuda()  # 1 x t x h x w

    instance_map = torch.zeros(time, height, width).byte()
    instances = []

    count = 1
    # mask = seed_map > 0.5
    mask = seed_map.bool()
    if mask.sum() > 128*time:
      spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(3, -1)
      sigma_masked = sigma[mask.expand_as(sigma)].view(n_sigma, -1)
      seed_map_masked = seed_map[mask].view(1, -1)

      unclustered = torch.ones(mask.sum()).byte().cuda()
      instance_map_masked = torch.zeros(mask.sum()).byte().cuda()

      # track used masks for computing iou
      used_ids = {}
      while (unclustered.sum() > 128*time):
        seed = (seed_map_masked * unclustered.float()).argmax().item()
        seed_score = (seed_map_masked * unclustered.float()).max().item()
        if seed_score < threshold:
          break
        center = spatial_emb_masked[:, seed:seed + 1].permute(1,0)
        unclustered[seed] = 0

        # calculate the precision matrix at seed point
        precision_vals = parse_embedding_output(sigma_masked[:, seed : seed + 1].permute(1, 0), embedding_size)
        precision_mat = precision_tensor_to_matrix(precision_vals, embedding_size)
        # print(sigma_masked[:, seed : seed + 1])
        assert precision_mat.shape[-2:] == (embedding_size, embedding_size)
        dist = torch.exp(-1 * mahalanobis_distance(spatial_emb_masked.permute(1, 0),
                                                   center=center, precision_mat=precision_mat,
                                                   return_squared_distance=True))
        # dist = dist.reshape(in_mask.shape[1:]).unsqueeze(0)

        proposal = (dist > PROBABILITY_THRESHOLD).squeeze()

        if proposal.sum() > 128*time:
          if unclustered[proposal].sum().float() / proposal.sum().float() > 0.5:
            instance_map_masked[proposal.squeeze()] = count
            instance_mask = torch.zeros(time, height, width).int()
            instance_mask[mask.squeeze().cpu()] = proposal.int().cpu()
            instances.append(
              {'mask': instance_mask.squeeze(), 'score': seed_score, 'centre': center})
            count += 1
            # calculate instance iou
            if iou_meter is not None and in_mask.shape[1] > 0:
              iou, id = get_best_overlap(instance_mask.numpy(),
                               in_mask.squeeze().data.cpu().numpy())
              if id not in used_ids.keys():
                used_ids[id] = calculate_iou(instance_mask.squeeze(), in_mask[id])
              elif iou > used_ids[id]:
                used_ids[id] = calculate_iou(instance_mask.squeeze(), in_mask[id])
              elif id == -1:
                iou_meter.update(0)
            elif iou_meter is not None and in_mask.shape[1] == 0:
              iou_meter.update(0)

        unclustered[proposal] = 0

      instance_map[mask.squeeze().cpu()] = instance_map_masked.cpu()
      if iou_meter is not None:
        if in_mask.shape[1] == 0 and len(instances == 0):
          iou_meter.update(1)
        else:
          iou_meter.update(np.nan_to_num(np.mean(list(used_ids.values()))))

    return instance_map, instances, None



class Logger:
  def __init__(self, keys, title=""):

    self.data = {k: [] for k in keys}
    self.title = title
    self.win = None

    print('created logger with keys:  {}'.format(keys))

  def plot(self, save=False, save_dir=""):

    if self.win is None:
      self.win = plt.subplots()
    fig, ax = self.win
    ax.cla()

    keys = []
    for key in self.data:
      keys.append(key)
      data = self.data[key]
      ax.plot(range(len(data)), data, marker='.')

    ax.legend(keys, loc='upper right')
    ax.set_title(self.title)

    plt.draw()
    Visualizer.mypause(0.001)

    if save:
      # save figure
      fig.savefig(os.path.join(save_dir, self.title + '.png'))

      # save data as csv
      df = pd.DataFrame.from_dict(self.data)
      df.to_csv(os.path.join(save_dir, self.title + '.csv'))

  def add(self, key, value):
    assert key in self.data, "Key not in data"
    self.data[key].append(value)
