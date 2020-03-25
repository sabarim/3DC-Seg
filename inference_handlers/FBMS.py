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
OVERLAPS = 7


def infer_fbms(dataset, model, criterion, writer, args, distributed=False):
  losses = AverageMeter()
  ious = AverageMeter()
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
      for iter, input_dict in enumerate(dataloader):
        if not args.exhaustive and (iter % (args.tw - OVERLAPS)) != 0:
          continue
        info = input_dict['info']
        if iter == 0:
          shape = tuple(info['num_frames'].int().numpy(), ) + tuple(input_dict['images'].shape[-2:], )
          # all_semantic_pred = torch.zeros(shape).int()

        batch_size = input_dict['images'].shape[0]
        input, input_var, loss, pred_dict = forward(criterion, input_dict, ious, model, args)
        losses.update(loss)
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


      ious.update(ious_per_video.avg)
      save_results(all_semantic_pred, info, os.path.join('results', args.network_name), palette)
      print('Sequence {}\t IOU {iou}'.format(input_dict['info']['name'], iou=ious_per_video.avg))

  print('Finished Inference Loss {losses.avg:.5f} IOU {iou.avg: 5f}'
        .format(losses=losses, iou=ious))


def save_results(pred, info, results_path, palette):
  results_path = os.path.join(results_path, info['name'][0])
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
    pickle.dump(prob, open('{:05d}.pkl'.format(f)))
