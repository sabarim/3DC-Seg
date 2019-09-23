import time
import numpy as np
import torch
import os

from PIL import Image
from scipy.misc import imresize
from torch.nn import functional as F

from inference_handlers.DAVIS import palette
from network.NetworkUtil import run_forward
from utils.AverageMeter import AverageMeter
from utils.Loss import bootstrapped_ce_loss
from utils.util import iou_fixed


def infer_DAVIS3d(dataloader, model, criterion, writer, args):
  batch_time = AverageMeter()
  losses = AverageMeter()
  ious = AverageMeter()
  results_path = os.path.join("results/", args.network_name)
  # switch to evaluate mode
  model.eval()
  end = time.time()
  iter = 0
  for seq in dataloader.dataset.get_video_ids():
  # for seq in ['blackswan']:
    dataloader.dataset.set_video_id(seq)
    ious_video = AverageMeter()
    all_preds = {}
    all_targets = {}

    for i, input_dict in enumerate(dataloader):
      info = input_dict['info']

      input, input_var, loss, pred = forward(criterion, input_dict, ious, model)
      clip_frames = info['support_indices'][0][1:].data.cpu().numpy() if args.test_dataset == "davis_siam" else \
        info['support_indices'][0].data.cpu().numpy()
      for i in range(len(clip_frames)):
        frame_id = clip_frames[i]
        if frame_id in all_preds:
          all_preds[frame_id] += [pred[0, :, i].data.cpu()]
        else:
          all_preds[frame_id] = [pred[0, :, i].data.cpu()]

        if frame_id not in all_targets:
          all_targets[frame_id] = input_dict['target'][0, 0, i].data.cpu()

    results = torch.stack([torch.stack(val).mean(dim=0) for key, val in all_preds.items()])
    targets = torch.stack([val for key, val in all_targets.items()])
    iou = iou_fixed(results.numpy(), targets.numpy())
    print('Sequence {}\t IOU {iou}'.format(input_dict['info']['name'], iou=iou))
    ious.update(iou, 1)
    save_results(results, info, os.path.join('results', args.network_name))

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
  if len(pred[0].shape) > 4:
    pred = F.interpolate(pred[0], target.shape[2:], mode="trilinear")
  else:
    pred = F.interpolate(pred[0], target.shape[2:], mode="bilinear")
  loss_image = criterion(pred[:, -1], target.squeeze(1).cuda().float())
  loss = bootstrapped_ce_loss(loss_image)

  return input, input_var, loss, pred


def save_results(pred, info, results_path):
  results_path = os.path.join(results_path, info['name'][0])
  pred = pred.data.cpu().numpy()
  # make hard label
  pred = np.argmax(pred, axis=1).astype(np.uint8)

  (lh, uh), (lw, uw) = info['pad']
  for f in range(len(pred)):
    E = pred[f, lh[0]:-uh[0], lw[0]:-uw[0]]


    img_E = Image.fromarray(imresize(E, info['shape'], interp='nearest'))
    img_E.putpalette(palette)
    if not os.path.exists(results_path):
      os.makedirs(results_path)
    img_E.save(os.path.join(results_path, '{:05d}.png'.format(f)))