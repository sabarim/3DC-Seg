import numpy as np
from torch.nn import functional as F

from loss.Loss import compute_loss
from network.NetworkUtil import run_forward
from utils.Constants import PRED_LOGITS, PRED_EMBEDDING, PRED_SEM_SEG
from utils.util import iou_fixed, iou_fixed_torch


def format_pred(pred):
  """

  :param pred: raw model predcitions
  :return: dict with formatted model predictions 
  """
  if type(pred) is not list:
    f_dict = {('%s' % PRED_LOGITS): pred}
  elif len(pred) == 1:
    f_dict = {PRED_LOGITS: pred[0]}
  elif len(pred) == 2:
    f_dict = {PRED_LOGITS: pred[0], PRED_EMBEDDING: pred[1]}
  elif len(pred) == 3:
    f_dict = {PRED_LOGITS: pred[0], PRED_SEM_SEG: pred[1], PRED_EMBEDDING: pred[2]}
  else:
    f_dict = None
  return f_dict


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
  pred = format_pred(pred)
  ious_extra = kwargs['ious_extra'] if 'ious_extra' in kwargs else None
  loss, loss_image, pred, loss_extra = compute_loss(args, criterion, pred, target, target_extra, iou_meter=ious_extra)
  pred = F.softmax(pred, dim=1)
  iou = iou_fixed_torch(pred, target.float().cuda())
  return iou, loss, loss_image, pred, loss_extra