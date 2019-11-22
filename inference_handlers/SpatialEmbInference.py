import os

import numpy as np
import torch
from PIL import Image
from scipy.misc import imresize
from torch.nn import functional as F

# cluster module
from torch.utils.data import DataLoader

from inference_handlers.infer_utils.CentreStitching import stitch_centres
from inference_handlers.infer_utils.LinearTubeStitching import stitch_clips_best_overlap
from inference_handlers.infer_utils.StitchWithGT import stitch_with_gt
from loss.Loss import compute_loss
from loss.SpatiotemporalEmbUtils import Cluster, Visualizer
from network.NetworkUtil import run_forward
from utils.AverageMeter import AverageMeter
from utils.Constants import DAVIS_ROOT

cluster = Cluster()
# number of overlapping frames for stitching
OVERLAPS = 0


def infer_spatial_emb(dataset, model, criterion, writer, args, distributed=False):
  losses = AverageMeter()
  ious = AverageMeter()
  # switch to evaluate mode
  model.eval()
  palette = Image.open(DAVIS_ROOT + '/Annotations_unsupervised/480p/bear/00000.png').getpalette()

  for seq in dataset.get_video_ids():
  # for seq in ['blackswan']:
    ious_per_video = AverageMeter()
    dataset.set_video_id(seq)
    test_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False) if distributed else None
    dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False, sampler=test_sampler, pin_memory=True)

    last_predictions = None
    stitched_instance_map = None
    all_instance_pred = None
    all_semantic_pred = None
    pred_mask_overlap = None
    for iter, input_dict in enumerate(dataloader):
      if not args.exhaustive and (iter % (args.tw - OVERLAPS)) != 0:
        continue
      info = input_dict['info']
      if iter == 0:
        shape = tuple(info['num_frames'].int().numpy(), ) + tuple(input_dict['images'].shape[-2:], )
        all_instance_pred = torch.zeros(shape).int()
        all_semantic_pred = torch.zeros(shape).int()

      target = input_dict['target_extra']['similarity_raw_mask']
      instances_one_hot = input_dict['target_extra']['similarity_ref'][:, 1:].squeeze()
      batch_size = input_dict['images'].shape[0]
      input, input_var, loss, pred, pred_extra = forward(criterion, input_dict, ious, model, args)
      losses.update(loss)
      pred_mask = F.softmax(pred, dim=1)
      #if pred_mask_overlap  is not None:
      #  pred_mask[:, -1, :OVERLAPS] = (pred_mask[:, -1, :OVERLAPS] + pred_mask_overlap.cuda()) / 2
      #pred_mask_overlap = pred_mask[:, -1, (pred_mask.shape[2] - OVERLAPS):].data
      seed_map = torch.argmax(pred_mask, dim=1).float() * pred_mask[:, -1]
      if args.embedding_dim - 4 == 3:
        seed_map = torch.argmax(pred_mask, dim=1).float() * pred_extra[:, -1]
        pred_extra[:, -1] = seed_map
        pred_spatemb = pred_extra.data.cpu()
      else:
        pred_spatemb = torch.cat((pred_extra.cuda(), seed_map.unsqueeze(1).float().cuda()), dim=1)
      instance_map, predictions = cluster.cluster(pred_spatemb[0], threshold=0.5, n_sigma=3,
                                                  iou_meter = ious_per_video, in_mask=instances_one_hot.data.cpu())

      assert batch_size == 1
      if args.stitch == 'gt':
        stitched_instance_map = stitch_with_gt(instance_map, instances_one_hot)
      elif stitched_instance_map is not None:
        stitched_instance_map = stitch_clips_best_overlap(stitched_instance_map, instance_map.data.cpu(), OVERLAPS)
      # if last_predictions is not None:
      #   stitched_instance_map, last_predictions = stitch_centres(ref_predictions=last_predictions,
      #                                                            curr_predictions=predictions,
      #                                                            tube_shape=instance_map.shape)
      #   all_instance_pred[info['support_indices'][0][OVERLAPS:]] = stitched_instance_map[OVERLAPS:].int()
      #   all_semantic_pred[info['support_indices'][0][OVERLAPS:]] = torch.argmax(pred_mask, dim=1).data.cpu().int()[0][OVERLAPS:]
      else:
        stitched_instance_map = instance_map.data.cpu()
        last_predictions = predictions
      all_instance_pred[info['support_indices'][0]] = stitched_instance_map.data.int().cpu()
      all_semantic_pred[info['support_indices'][0]] = torch.argmax(pred_mask, dim=1).data.cpu().int()[0]
      # for i in range(input_dict['images'].shape[2]):
      #   visualise(input_dict, instance_map, target, pred_spatemb, i, args, iter)
        # save_results(input_dict['images'], predictions, info, args, i, iter)
      #all_instance_pred[info['support_indices'][0]] = stitched_instance_map.int()
      #all_semantic_pred[info['support_indices'][0]] = torch.argmax(pred_mask, dim=1).data.cpu().int()[0]


    ious.update(ious_per_video.avg)
    save_results(all_instance_pred, info, os.path.join('results', args.network_name), palette)
    save_results(all_semantic_pred, info, os.path.join('results', args.network_name, 'semantic_pred'), palette)
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
  instances = instances.squeeze()[i]
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


def forward(criterion, input_dict, ious, model, args):
  input = input_dict["images"]
  target = input_dict["target"]
  target_extra = None if 'target_extra' not in input_dict else input_dict['target_extra']
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
  # loss, loss_image, _, loss_extra = compute_loss(args, criterion, (pred, pred_extra), target, target_extra,
  #                                                iou_meter=AverageMeter())
  loss=0

  return input, input_var, loss, pred, pred_extra


def save_results(pred, info, results_path, palette):
  results_path = os.path.join(results_path, info['name'][0])
  pred = pred.data.cpu().numpy().astype(np.uint8)
  (lh, uh), (lw, uw) = info['pad']
  h, w = pred.shape[-2:]
  for f in range(len(pred)):
    M = pred[f, lh[0]:h-uh[0], lw[0]:w-uw[0]]
    img_M = Image.fromarray(imresize(M, info['shape'], interp='nearest'))
    img_M.putpalette(palette)
    if not os.path.exists(results_path):
      os.makedirs(results_path)
    img_M.save(os.path.join(results_path, '{:05d}.png'.format(f)))


