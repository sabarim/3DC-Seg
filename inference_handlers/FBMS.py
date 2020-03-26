import os
import pickle

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

cluster = Cluster()
# number of overlapping frames for stitching
OVERLAPS = 3


def infer_fbms(dataset, model, criterion, writer, args, distributed=False):
  Fs = AverageMeter()
  maes = AverageMeter()
  # switch to evaluate mode
  model.eval()
  palette = Image.open(DAVIS_ROOT + '/Annotations_unsupervised/480p/bear/00000.png').getpalette()

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
        ious_per_video.update()

        assert batch_size == 1
        for i, f in enumerate(clip_frames):
          if f in all_semantic_pred:
            # all_semantic_pred[clip_frames] += [torch.argmax(pred_mask, dim=1).data.cpu().int()[0]]
            all_semantic_pred[f] += [pred_mask[0, :, i].data.cpu().float()]
          else:
            all_semantic_pred[f] = [pred_mask[0, :, i].data.cpu().float()]
            if f in info['gt_frames']:
              all_targets[f] = [target[0, i].data.cpu().float()]



      F, mae = save_results(all_semantic_pred, all_targets, info, os.path.join('results', args.network_name), palette)
      Fs.update(F)
      maes.update(mae)
      print('Sequence {}: F_max {}  MAE {}'.format(input_dict['info']['name'], F, mae))

  print('Finished Inference F measure: {Fs.avg:.5f} MAE: {maes.avg: 5f}'
        .format(Fs=Fs, maes=maes))


def save_results(pred, targets, info, results_path, palette):
  results_path = os.path.join(results_path, info['name'][0])
  pred_for_eval = []
  # pred = pred.data.cpu().numpy().astype(np.uint8)
  (lh, uh), (lw, uw) = info['pad']
  for f in pred.keys():
    M = torch.argmax(torch.stack(pred[f]).mean(dim=0), dim=0)
    h, w = M.shape[-2:]
    M = M[lh[0]:h-uh[0], lw[0]:w-uw[0]]
    shape = info['shape480p'] if 'shape480p' in info else info['shape']
    img_M = Image.fromarray(imresize(M, shape, interp='nearest'))
    img_M.putpalette(palette)
    if not os.path.exists(results_path):
      os.makedirs(results_path)
    img_M.save(os.path.join(results_path, '{:05d}.png'.format(f)))
    prob = torch.stack(pred[f]).mean(dim=0)[-1]
    pickle.dump(prob, open('{:05d}.pkl'.format(f), 'wb'))
    if f in targets:
      pred_for_eval += [torch.stack(pred[f]).mean(dim=0)]

  assert len(targets.values()) == len(pred_for_eval)
  pred_for_F = np.stack(torch.argmax(pred_for_eval, dim=0)).flatten()
  pred_for_mae = np.stack(pred_for_eval[-1]).flatten()
  gt = np.stack(targets.values()).flatten()
  precision, recall, _= precision_recall_curve(gt.data.cpu().numpy(), pred_for_F.data.cpu().numpy())
  Fmax = 2 * (precision * recall) / (precision + recall)
  mae = np.mean(abs(pred_for_mae - gt))

  return Fmax, mae
