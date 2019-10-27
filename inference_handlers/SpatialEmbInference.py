import os

import numpy as np
import torch
from PIL import Image
from scipy.misc import imresize
from torch.nn import functional as F

from inference_handlers.DAVIS import palette
# cluster module
from inference_handlers.infer_utils.CentreStitching import stitch_centres
from inference_handlers.infer_utils.LinearTubeStitching import stitch_clips_best_overlap
from loss.SpatiotemporalEmbUtils import Cluster, Visualizer
from network.NetworkUtil import run_forward
from utils.AverageMeter import AverageMeter

cluster = Cluster()
# number of overlapping frames for stitching
OVERLAPS = 2


def infer_spatial_emb(dataloader, model, criterion, writer, args):
  losses = AverageMeter()
  ious = AverageMeter()
  # switch to evaluate mode
  model.eval()

  # for seq in dataloader.dataset.get_video_ids():
  for seq in ['breakdance']:
    ious_per_video = AverageMeter()
    dataloader.dataset.set_video_id(seq)
    last_predictions = None
    all_instance_pred = None
    all_semantic_pred = None
    pred_mask_overlap = None
    for iter, input_dict in enumerate(dataloader):
      if not args.exhaustive and (iter % (args.tw - OVERLAPS)) != 0:
        continue
      last_predictions = last_predictions
      info = input_dict['info']
      if iter == 0:
        shape = tuple(info['num_frames'].int().numpy(), ) + tuple(input_dict['images'].shape[-2:], )
        all_instance_pred = torch.zeros(shape).int()
        all_semantic_pred = torch.zeros(shape).int()

      instances_one_hot = input_dict['target_extra']['similarity_ref'][:, 1:].squeeze()
      batch_size = input_dict['images'].shape[0]
      input, input_var, loss, pred, pred_extra = forward(criterion, input_dict, ious, model)
      pred_mask = F.softmax(pred, dim=1)
      if pred_mask_overlap  is not None:
        pred_mask[:, -1, :OVERLAPS] = (pred_mask[:, -1, :OVERLAPS] + pred_mask_overlap.cuda()) / 2
      pred_mask_overlap = pred_mask[:, -1, (pred_mask.shape[2] - OVERLAPS):].data
      seed_map = torch.argmax(pred_mask, dim=1).float() * pred_mask[:, -1]
      if args.embedding_dim - 4 == 3:
        # seed_map = torch.argmax(pred_mask, dim=1).float() * pred_extra[:, -1]
        pred_extra[:, -1] = seed_map
        pred_spatemb = pred_extra
      else:
        pred_spatemb = torch.cat((pred_extra.cuda(), seed_map.unsqueeze(1).float().cuda()), dim=1)
      instance_map, predictions = cluster.cluster(pred_spatemb[0], threshold=0.5, n_sigma=3,
                                                  iou_meter = ious_per_video, in_mask=instances_one_hot)

      assert batch_size == 1
      if last_predictions is not None:
        # stitched_instance_map = stitch_clips_best_overlap(last_predictions, instance_map, OVERLAPS)
        stitched_instance_map, last_predictions = stitch_centres(ref_predictions=last_predictions,
                                                                 curr_predictions=predictions,
                                                                 tube_shape=instance_map.shape)
      else:
        stitched_instance_map = instance_map
        last_predictions = predictions
      # for i in range(input_dict['images'].shape[2]):
        # visualise(input_dict, instance_map, instances, pred_spatemb, i, args, iter)
        # save_results(input_dict['images'], predictions, info, args, i, iter)
      all_instance_pred[info['support_indices'][0]] = stitched_instance_map.int()
      all_semantic_pred[info['support_indices'][0]] = torch.argmax(pred_mask, dim=1).data.cpu().int()[0]


    ious.update(ious_per_video.avg)
    save_results(all_instance_pred, info, os.path.join('results', args.network_name))
    save_results(all_semantic_pred, info, os.path.join('results', args.network_name, 'semantic_pred'))
    print('Sequence {}\t IOU {iou}'.format(input_dict['info']['name'], iou=ious_per_video.avg))

  print('Finished Inference Loss {losses.avg:.5f} IOU {iou.avg: 5f}'
        .format(losses=losses, iou=ious))


def visualise(input_dict, instance_map, instances, pred_extra, i, args, iter):
  # Visualizer
  visualizer = Visualizer(('image', 'pred', 'sigma', 'seed'))
  results_path = os.path.join('results', args.network_name, input_dict['info']['name'][0])
  if not os.path.exists(results_path):
    os.makedirs(results_path)
  base, _ = os.path.splitext(os.path.basename(input_dict['info']['name'][0]))

  im = input_dict['images'][0, :, i]
  instances = instances[i]
  instance_map = instance_map[i]
  # visualizer.display(im, 'image')
  visualizer.savePlt([instance_map.cpu(), instances.cpu()], 'pred', os.path.join(results_path, base +
                                                                                 '_{:05d}_pred.png'.format(iter+i)))
  sigma = pred_extra[0, 3:-1, i].data.cpu()
  sigma = (sigma - sigma.min()) / (sigma.max() - sigma.min())
  sigma[:, instances == 0] = 0
  count=0
  for s in sigma:
    count+=1
    visualizer.savePlt(s.cpu(), 'sigma', os.path.join(results_path, base +
                                                          '_{:05d}_sigma_{:1d}.png'.format(iter+i, count)))
  seed = torch.sigmoid(pred_extra[0, -1, i].data.cpu())
  visualizer.savePlt(seed, 'seed', os.path.join(results_path, base +
                                                '_{:05d}_seed.png'.format(iter+i)))


def forward(criterion, input_dict, ious, model):
  input = input_dict["images"]
  target = input_dict["target"]
  if 'masks_guidance' in input_dict:
    masks_guidance = input_dict["masks_guidance"]
    masks_guidance = masks_guidance.float().cuda()
  else:
    masks_guidance = None
  info = input_dict["info"]
  # data_time.update(time.time() - end)
  input_var = input.float().cuda()
  # compute output
  pred = run_forward(model, input_var, masks_guidance, input_dict['proposals'])
  pred_extra = pred[1] if len(pred) > 1 else None
  if len(pred[0].shape) > 4:
    pred = F.interpolate(pred[0], target.shape[2:], mode="trilinear")
    if pred_extra is not None:
      pred_extra = F.interpolate(pred_extra, target.shape[2:], mode="trilinear")
  else:
    pred = F.interpolate(pred[0], target.shape[2:], mode="bilinear")
    pred_extra = F.interpolate(pred_extra, target.shape[2:], mode="bilinear")
  loss = 0

  return input, input_var, loss, pred, pred_extra


def save_results(pred, info, results_path):
  results_path = os.path.join(results_path, info['name'][0])
  pred = pred.data.cpu().numpy().astype(np.uint8)
  (lh, uh), (lw, uw) = info['pad']
  for f in range(len(pred)):
    M = pred[f, lh[0]:-uh[0], lw[0]:-uw[0]]
    img_M = Image.fromarray(imresize(M, info['shape'], interp='nearest'))
    img_M.putpalette(palette)
    if not os.path.exists(results_path):
      os.makedirs(results_path)
    img_M.save(os.path.join(results_path, '{:05d}.png'.format(f)))


