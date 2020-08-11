import logging
import os
import pickle

import numpy as np
from abc import abstractmethod

import torch
from PIL import Image
from scipy.misc import imresize
from sklearn.metrics import precision_recall_curve
from torch.utils.data import DataLoader
from torch.nn import functional as F

from util import color_map
from utils.AverageMeter import AverageMeter
from utils.Constants import PRED_LOGITS, PRED_SEM_SEG
from utils.util import iou_fixed_torch


class BaseInferenceEngine():
  def __init__(self, cfg):
    self.cfg = cfg
    self.results_dir = os.path.join('results', cfg.NAME)
    if not os.path.exists(self.results_dir):
      os.makedirs(self.results_dir)
    log_file = os.path.join(self.results_dir, 'output.log')
    logging.basicConfig(filename=log_file, level=logging.INFO)

  def infer(self, dataset, model):
    pass
  
  
class SaliencyInferenceEngine(BaseInferenceEngine):
  def __init__(self, cfg):
    super(SaliencyInferenceEngine, self).__init__(cfg)

  def infer(self, dataset, model):
    fs = AverageMeter()
    maes = AverageMeter()
    ious = AverageMeter()
    # switch to evaluate mode
    model.eval()
    pred_for_eval = []
    gt_for_eval = []

    with torch.no_grad():
      for seq in dataset.get_video_ids():
        ious_per_video = AverageMeter()
        dataset.set_video_id(seq)
        # test_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False) if distributed else None
        test_sampler = None
        dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False, sampler=test_sampler,
                                pin_memory=True)

        all_semantic_pred = {}
        all_targets = {}
        for iter, input_dict in enumerate(dataloader):
          if not self.cfg.INFERENCE.EXHAUSTIVE and (iter % (self.cfg.INPUT.TW - self.cfg.INFERENCE.CLIP_OVERLAP)) != 0:
            continue

          info = input_dict['info']
          input = input_dict["images"]
          batch_size = input.shape[0]
          target_dict = dict([(k, t.float().cuda()) for k, t in input_dict['target'].items()])
          input_var = input.float().cuda()

          # compute output
          pred = model(input_var)
          # pred = format_pred(pred)

          pred_mask = F.softmax(pred[0], dim=1)
          clip_frames = info[0]['support_indices'][0].data.cpu().numpy()

          assert batch_size == 1
          for i, f in enumerate(clip_frames):
            if f in all_semantic_pred:
              # all_semantic_pred[clip_frames] += [torch.argmax(pred_mask, dim=1).data.cpu().int()[0]]
              all_semantic_pred[f] += [pred_mask[0, :, i].data.cpu().float()]
            else:
              all_semantic_pred[f] = [pred_mask[0, :, i].data.cpu().float()]
              # Use binary masks
              all_targets[f] = (target_dict['mask'] != 0)[0, 0, i].data.cpu().float()

        masks = [torch.stack(pred).mean(dim=0) for key, pred in all_semantic_pred.items() if key in all_targets]
        iou = iou_fixed_torch(torch.stack(masks).cuda(), torch.stack(list(all_targets.values())).cuda())
        ious_per_video.update(iou, 1)
        ious.update(iou, 1)
        f, mae, pred_flattened, gt_flattened = self.save_results(all_semantic_pred, all_targets, info)
        fs.update(f)
        maes.update(mae)
        pred_for_eval += [pred_flattened]
        gt_for_eval += [gt_flattened]
        logging.info(
          'Sequence {}: F_max {}  MAE {} IOU {}'.format(input_dict['info'][0]['video'], f, mae, ious_per_video.avg))

    print("IOU: {}".format(iou))
    gt = np.hstack(gt_for_eval).flatten()
    p = np.hstack(pred_for_eval).flatten()
    precision, recall, _ = precision_recall_curve(gt, p)
    Fmax = 2 * (precision * recall) / (precision + recall)
    mae = np.mean(np.abs(p - gt))
    logging.info('Finished Inference F measure: {:.5f} MAE: {: 5f} IOU: {:5f}'
                 .format(np.max(Fmax), mae, ious.avg))

  def save_results(self, pred, targets, info):
    results_path = os.path.join(self.results_dir, info[0]['video'][0])
    pred_for_eval = []
    # pred = pred.data.cpu().numpy().astype(np.uint8)
    (lh, uh), (lw, uw) = info[0]['pad']
    for f in pred.keys():
      M = torch.argmax(torch.stack(pred[f]).mean(dim=0), dim=0)
      h, w = M.shape[-2:]
      M = M[lh[0]:h - uh[0], lw[0]:w - uw[0]]

      if f in targets:
        pred_for_eval += [torch.stack(pred[f]).mean(dim=0)[:, lh[0]:h - uh[0], lw[0]:w - uw[0]]]

      shape = info[0]['shape']
      img_M = Image.fromarray(imresize(M.byte(), shape, interp='nearest'))
      img_M.putpalette(color_map().flatten().tolist())
      if not os.path.exists(results_path):
        os.makedirs(results_path)
      img_M.save(os.path.join(results_path, '{:05d}.png'.format(f)))
      if self.cfg.INFERENCE.SAVE_LOGITS:
        prob = torch.stack(pred[f]).mean(dim=0)[-1]
        pickle.dump(prob, open(os.path.join(results_path, '{:05d}.pkl'.format(f)), 'wb'))

    assert len(targets.values()) == len(pred_for_eval)
    pred_for_F = torch.argmax(torch.stack(pred_for_eval), dim=1)
    pred_for_mae = torch.stack(pred_for_eval)[:, -1]
    gt = torch.stack(list(targets.values()))[:, lh[0]:h - uh[0], lw[0]:w - uw[0]]
    precision, recall, _ = precision_recall_curve(gt.data.cpu().numpy().flatten(),
                                                  pred_for_F.data.cpu().numpy().flatten())
    Fmax = 2 * (precision * recall) / (precision + recall)
    mae = (pred_for_mae - gt).abs().mean()

    return Fmax.max(), mae, pred_for_mae.data.cpu().numpy().flatten(), gt.data.cpu().numpy().flatten()

