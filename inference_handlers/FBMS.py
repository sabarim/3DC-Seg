import os
import pickle
import logging
import numpy as np
import torch
from PIL import Image
from scipy.misc import imresize
from sklearn.metrics import precision_recall_curve
from torch.nn import functional as F
# cluster module
from torch.utils.data import DataLoader

from inference_handlers.SpatialEmbInference import forward
from loss.SpatiotemporalEmbUtils import Cluster
from utils.AverageMeter import AverageMeter
from utils.Constants import DAVIS_ROOT, PRED_LOGITS, PRED_SEM_SEG
from utils.util import iou_fixed_torch

cluster = Cluster()
# number of overlapping frames for stitching
OVERLAPS = 3


def infer_fbms(dataset, model, criterion, writer, args, distributed=False):
  fs = AverageMeter()
  maes = AverageMeter()
  ious = AverageMeter()
  # switch to evaluate mode
  model.eval()
  palette = Image.open(DAVIS_ROOT + '/Annotations_unsupervised/480p/bear/00000.png').getpalette()
  results_dir = os.path.join('results', args.network_name)
  if not os.path.exists(results_dir):
      os.makedirs(results_dir)
  log_file = os.path.join(results_dir, 'output.log')
  logging.basicConfig(filename=log_file,level=logging.DEBUG)

  with torch.no_grad():
    for seq in dataset.get_video_ids():
      ious_per_video = AverageMeter()
      dataset.set_video_id(seq)
      test_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False) if distributed else None
      dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False, sampler=test_sampler, pin_memory=True)

      all_semantic_pred = {}
      all_targets = {}
      for iter, input_dict in enumerate(dataloader):
        if not args.exhaustive and (iter % (args.tw - OVERLAPS)) != 0:
          continue
        info = input_dict['info']
        target = input_dict['target']
        if iter == 0:
          shape = tuple(info['num_frames'].int().numpy(), ) + tuple(input_dict['images'].shape[-2:], )
          # all_semantic_pred = torch.zeros(shape).int()

        batch_size = input_dict['images'].shape[0]
        input, input_var, loss, pred_dict = forward(criterion, input_dict, ious, model, args)
        pred_mask = F.softmax(pred_dict[PRED_LOGITS], dim=1)
        pred_multi = F.softmax(pred_dict[PRED_SEM_SEG], dim=1) if PRED_SEM_SEG in pred_dict else None

        clip_frames = info['support_indices'][0].data.cpu().numpy()

        assert batch_size == 1
        for i, f in enumerate(clip_frames):
          if f in all_semantic_pred:
            # all_semantic_pred[clip_frames] += [torch.argmax(pred_mask, dim=1).data.cpu().int()[0]]
            all_semantic_pred[f] += [pred_mask[0, :, i].data.cpu().float()]
          else:
            all_semantic_pred[f] = [pred_mask[0, :, i].data.cpu().float()]
            if 'gt_frames' not in info or f in info['gt_frames']:
              all_targets[f] = target[0, 0, i].data.cpu().float()

      masks = [torch.stack(pred).mean(dim=0) for pred in all_semantic_pred.values()]
      iou = iou_fixed_torch(torch.stack(masks), torch.stack(list(all_targets.values())))
      ious_per_video.update(iou, 1)
      f, mae = save_results(all_semantic_pred, all_targets, info, os.path.join('results', args.network_name), palette)
      fs.update(f)
      maes.update(mae)
      logging.info('Sequence {}: F_max {}  MAE {} IOU {}'.format(input_dict['info']['name'], f, mae, ious_per_video.avg))

  logging.info('Finished Inference F measure: {fs.avg:.5f} MAE: {maes.avg: 5f}'
        .format(fs=fs, maes=maes))


def save_results(pred, targets, info, results_path, palette):
  results_path = os.path.join(results_path, info['name'][0])
  pred_for_eval = []
  # pred = pred.data.cpu().numpy().astype(np.uint8)
  (lh, uh), (lw, uw) = info['pad']
  for f in pred.keys():
    M = torch.argmax(torch.stack(pred[f]).mean(dim=0), dim=0)
    h, w = M.shape[-2:]
    M = M[lh[0]:h-uh[0], lw[0]:w-uw[0]]

    if f in targets:
        pred_for_eval += [torch.stack(pred[f]).mean(dim=0)[:, lh[0]:h-uh[0], lw[0]:w-uw[0]]]

    shape = info['shape']
    img_M = Image.fromarray(imresize(M.byte(), shape, interp='nearest'))
    img_M.putpalette(palette)
    if not os.path.exists(results_path):
      os.makedirs(results_path)
    img_M.save(os.path.join(results_path, '{:05d}.png'.format(f)))
    prob = torch.stack(pred[f]).mean(dim=0)[-1]
    pickle.dump(prob, open(os.path.join(results_path, '{:05d}.pkl'.format(f)), 'wb'))

  assert len(targets.values()) == len(pred_for_eval)
  pred_for_F = torch.argmax(torch.stack(pred_for_eval), dim=1)
  pred_for_mae = torch.stack(pred_for_eval)[:, -1]
  gt = torch.stack(list(targets.values()))[:, lh[0]:h-uh[0], lw[0]:w-uw[0]]
  precision, recall, _= precision_recall_curve(gt.data.cpu().numpy().flatten(), pred_for_F.data.cpu().numpy().flatten())
  Fmax = 2 * (precision * recall) / (precision + recall)
  mae = (pred_for_mae - gt).abs().mean()

  return Fmax.max(), mae
