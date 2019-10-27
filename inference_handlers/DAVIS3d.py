import os
import pickle
import time

import numpy as np
import torch
from PIL import Image
from scipy.misc import imresize
from torch.nn import functional as F

from inference_handlers.DAVIS import palette
from network.NetworkUtil import run_forward
from utils.AverageMeter import AverageMeter
from utils.util import iou_fixed


def infer_DAVIS3d(dataloader, model, criterion, writer, args):
  batch_time = AverageMeter()
  losses = AverageMeter()
  ious = AverageMeter()
  results_path = os.path.join("results/", args.network_name)
  # switch to evaluate mode
  model.eval()
  end = time.time()
  for seq in dataloader.dataset.get_video_ids():
  # for seq in ['horsejump-high']:
    dataloader.dataset.set_video_id(seq)
    ious_video = AverageMeter()
    all_preds = {}
    all_targets = {}
    all_e = {}

    for iter, input_dict in enumerate(dataloader):
      if not args.exhaustive and iter % args.tw != 0:
        continue
      info = input_dict['info']
      input, input_var, loss, pred, pred_extra = forward(criterion, input_dict, ious, model)
      clip_frames = info['support_indices'][0][1:].data.cpu().numpy() if args.test_dataset == "davis_siam" else \
        info['support_indices'][0].data.cpu().numpy()
      for i in range(len(clip_frames)):
        frame_id = clip_frames[i]
        if frame_id in all_preds:
          all_preds[frame_id] += [pred[0, :, i].data.cpu()]
          all_e[frame_id] += [pred_extra[0, :, i].data.cpu()]
        else:
          all_preds[frame_id] = [pred[0, :, i].data.cpu()]
          all_e[frame_id] = [pred_extra[0, :, i].data.cpu()]

        if frame_id not in all_targets:
          all_targets[frame_id] = input_dict['target'][0, 0, i].data.cpu()
      if args.save_per_clip and pred_extra is not None:
        save_per_clip(iter, pred_extra.data.cpu(), info, os.path.join('results', args.network_name))


    results = torch.stack([torch.stack(val).mean(dim=0) for key, val in all_preds.items()])
    targets = torch.stack([val for key, val in all_targets.items()])
    iou = iou_fixed(results.numpy(), targets.numpy())
    print('Sequence {}\t IOU {iou}'.format(input_dict['info']['name'], iou=iou))
    ious.update(iou, 1)

    results_extra = torch.stack([torch.stack(val).mean(dim=0) for key, val in all_e.items()])
    save_results(results, results_extra.permute(1,0,2,3), info, os.path.join('results', args.network_name))

  print('Finished Inference Loss {losses.avg:.5f} IOU {iou.avg: 5f}'
        .format(losses=losses, iou=ious))


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
  loss=0

  return input, input_var, loss, F.softmax(pred, dim=1), pred_extra


def save_results(pred, pred_extra, info, results_path):
  results_path = os.path.join(results_path, info['name'][0])
  pred = pred.data.cpu().numpy()
  # make hard label
  pred = np.argmax(pred, axis=1).astype(np.uint8)
  e_path = os.path.join(results_path, 'embeddings')
  if not os.path.exists(e_path):
    os.makedirs(e_path)

  (lh, uh), (lw, uw) = info['pad']
  for f in range(len(pred)):
    M = pred[f, lh[0]:-uh[0], lw[0]:-uw[0]]
    img_M = Image.fromarray(imresize(M, info['shape'], interp='nearest'))
    img_M.putpalette(palette)
    if not os.path.exists(results_path):
      os.makedirs(results_path)
    img_M.save(os.path.join(results_path, '{:05d}.png'.format(f)))

  # e = pred_extra[:, :, lh[0]:-uh[0], lw[0]:-uw[0]] if pred_extra is not None else None
  # e_dict = {"embeddings": e, 'frames': list(range(len(pred)))}
  # if e is not None:
  #   e.numpy().dump(os.path.join(results_path, '{:05d}.pickle'.format(f)))
  # with open(os.path.join(e_path, 'clip_{:05d}_{:05d}.pickle'.format(0, len(pred))), 'wb') as f:
  #   pickle.dump(e_dict, f)


def save_per_clip(iter, e, info, results_path):
  results_path = os.path.join(results_path, info['name'][0])
  e_path = os.path.join(results_path, 'embeddings')
  if not os.path.exists(e_path):
    os.makedirs(e_path)

  (lh, uh), (lw, uw) = info['pad']
  e = e[:, :, :, lh[0]:-uh[0], lw[0]:-uw[0]]
  # o = io.BytesIO()
  # compressed_e = np.savez_compressed(o, e.numpy())
  save_dict = {"embeddings": e, 'frames': info['support_indices'][0].data.cpu()}
  with open(os.path.join(e_path, 'clip_{:05d}_{:05d}.pickle'.format(iter, iter + 7)), 'wb') as f:
    pickle.dump(save_dict, f)
  # np.savez_compressed(f, a=e.numpy(), b=info['support_indices'][0].data.cpu().numpy())
