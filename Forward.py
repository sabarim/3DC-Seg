import numpy as np
from torch.nn import functional as F

from loss.Loss import compute_loss
from network.NetworkUtil import run_forward
from utils.util import iou_fixed, iou_fixed_torch


def forward(args, criterion, input_dict, model, **kwargs):
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
  ious_extra = kwargs['ious_extra'] if 'ious_extra' in kwargs else None
  loss, loss_image, pred, loss_extra = compute_loss(args, criterion, pred, target, target_extra, iou_meter=ious_extra)
  pred = F.softmax(pred, dim=1)
  iou = iou_fixed_torch(pred, target.float().cuda())
  return iou, loss, loss_image, pred, loss_extra