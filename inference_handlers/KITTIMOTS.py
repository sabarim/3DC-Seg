import os
import numpy as np
import torch
from PIL import Image
from scipy.misc import imresize
from torch.utils.data import DataLoader
from torch.nn import functional as F

from datasets.coco.COCO import COCO_SUPERCATEGORIES
from inference_handlers.SpatialEmbInference import forward, cluster, save_per_clip
from inference_handlers.infer_utils.LinearTubeStitching import stitch_clips_best_overlap
from inference_handlers.infer_utils.StitchWithGT import stitch_with_gt
from inference_handlers.infer_utils.output_utils import get_class_label
from scripts.path_constants import DAVIS_ROOT
from utils.AverageMeter import AverageMeter
from utils.Constants import PRED_LOGITS, PRED_SEM_SEG, PRED_EMBEDDING
from utils.util import get_iou

OVERLAPS = 0


def infer_kitti_mots(dataset, model, criterion, writer, args, distributed=False):
  losses = AverageMeter()
  ious = AverageMeter()
  # switch to evaluate mode
  model.eval()
  palette = Image.open(DAVIS_ROOT + '/Annotations_unsupervised/480p/bear/00000.png').getpalette()

  with torch.no_grad():
    for seq in dataset.get_video_ids():
      # for seq in ['lab-coat']:
      ious_per_video = AverageMeter()
      dataset.set_video_id(seq)
      test_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False) if distributed else None
      dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False, sampler=test_sampler,
                              pin_memory=True)

      stitched_instance_map = None
      all_instance_pred = None
      all_semantic_pred = None
      for iter, input_dict in enumerate(dataloader):
        if not args.exhaustive and (iter % (args.tw - OVERLAPS)) != 0:
          continue
        info = input_dict['info']
        if iter == 0:
          shape = tuple(info['num_frames'].int().numpy(), ) + tuple(input_dict['images'].shape[-2:], )
          all_instance_pred = torch.zeros(shape).int()
          all_semantic_pred = torch.zeros(shape).int()

        target = input_dict['target_extra']['similarity_raw_mask']
        instances_one_hot = get_one_hot(target)
        batch_size = input_dict['images'].shape[0]
        input, input_var, loss, pred_dict = forward(criterion, input_dict, ious, model, args)
        losses.update(loss)
        pred_mask = F.softmax(pred_dict[PRED_LOGITS], dim=1)
        pred_multi = F.softmax(pred_dict[PRED_SEM_SEG], dim=1)
        pred_embedding = pred_dict[PRED_EMBEDDING]

        seed_map = torch.argmax(pred_mask, dim=1).float() * torch.sigmoid(pred_embedding[:, -1])
        pred_embedding[:, -1] = seed_map
        pred_spatemb = pred_embedding.data.cpu()

        instance_map, predictions = cluster.cluster(pred_spatemb[0], threshold=0.5, n_sigma=args.n_sigma,
                                                    iou_meter=ious_per_video, in_mask=instances_one_hot.data.cpu())

        clip_frames = info['support_indices'][0].data.cpu().numpy()
        if args.save_per_clip and pred_embedding is not None:
          save_per_clip(iter, instance_map=instance_map, e=None, info=info,
                        results_path=os.path.join('results', args.network_name, 'per_clip'), palette=palette)
          if pred_multi is not None:
            save_per_clip(iter, instance_map=torch.argmax(pred_multi[0].data.cpu(), dim=0), e=None, info=info,
                          results_path=os.path.join('results', args.network_name, 'sem_seg'), palette=palette)

        assert batch_size == 1
        if args.stitch == 'gt':
          stitched_instance_map = stitch_with_gt(instance_map, target)
          all_instance_pred[clip_frames] = stitched_instance_map.data.int().cpu()
          all_semantic_pred[clip_frames] = torch.argmax(pred_multi, dim=1).data.cpu().int()[0]
        elif stitched_instance_map is not None:
          stitched_instance_map = stitch_clips_best_overlap(stitched_instance_map, instance_map.data.cpu(), OVERLAPS)
          all_instance_pred[clip_frames[OVERLAPS:]] = stitched_instance_map.data.int().cpu()[OVERLAPS:].int()
          all_semantic_pred[clip_frames[OVERLAPS:]] = torch.argmax(pred_multi, dim=1).data.cpu().int()[0][OVERLAPS:]
        else:
          stitched_instance_map = instance_map.data.cpu()
          all_instance_pred[clip_frames] = stitched_instance_map.data.int().cpu()
          all_semantic_pred[clip_frames] = torch.argmax(pred_multi, dim=1).data.cpu().int()[0]

      ious.update(ious_per_video.avg)
      save_results(all_instance_pred, all_semantic_pred, info, os.path.join('results', args.network_name), palette)
      # save_results(all_semantic_pred, info, os.path.join('results', args.network_name, 'semantic_pred'), palette)
      print('Sequence {}\t IOU {iou}'.format(input_dict['info']['name'], iou=ious_per_video.avg))

  print('Finished Inference Loss {losses.avg:.5f} IOU {iou.avg: 5f}'
        .format(losses=losses, iou=ious))


def get_one_hot(mask):
  objects = mask.unique()
  one_hot_mask = torch.zeros((max(objects) + 1,) + mask.shape[-3:], dtype=torch.int)

  for i in objects:
    one_hot_mask[i] = (mask == i).int()

  return one_hot_mask[1:]


def save_results(pred, classes, info, results_path, palette):
  results_path_mots = os.path.join(results_path, 'mots', info['name'][0])
  results_path = os.path.join(results_path, info['name'][0])
  pred = pred.data.cpu().numpy().astype(np.uint8)
  instances = np.setdiff1d(np.unique(pred), [0])
  results = np.zeros_like(pred).astype(np.uint16)
  coco_cats_mapping = {COCO_SUPERCATEGORIES.index('vehicle') + 1 : 1, COCO_SUPERCATEGORIES.index('person') + 1 : 2}
  for instance in instances:
    label = get_class_label((pred == instance), classes)
    label = coco_cats_mapping[label] if label in coco_cats_mapping else label
    results[pred == instance] = 1000 * label + instance

  (lh, uh), (lw, uw) = info['pad']
  h, w = pred.shape[-2:]
  for f in range(len(pred)):
    M_mots = results[f, lh[0]:h - uh[0], lw[0]:w - uw[0]]
    M = pred[f, lh[0]:h - uh[0], lw[0]:w - uw[0]]
    img_M = Image.fromarray(M)
    img_M.putpalette(palette)

    img_mots = Image.fromarray(M_mots)

    if not os.path.exists(results_path):
      os.makedirs(results_path)
    if not os.path.exists(results_path_mots):
      os.makedirs(results_path_mots)
    img_M.save(os.path.join(results_path, '{:05d}.png'.format(f)))
    img_mots.save(os.path.join(results_path_mots, '{:05d}.png'.format(f)))
