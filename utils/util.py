import torch
import numpy as np


def ToOneHot(labels, num_objects):
  print(labels)
  labels = labels.view(-1, 1)
  labels = torch.eye(num_objects).index_select(dim=0, index=labels)
  return labels.cuda()


def ToLabel(E):
  fgs = np.argmax(E, axis=1).astype(np.float32)
  return fgs.astype(np.uint8)


def iou_fixed(pred, gt, exclude_last=False):
  pred = ToLabel(pred)
  ious = []
  num_frames = pred.shape[0]
  end = num_frames
  if exclude_last:
    end -= 1
  for t in range(0, end):
    i = np.logical_and(pred[t] > 0, gt[t] > 0).sum()
    u = np.logical_or(pred[t] > 0, gt[t] > 0).sum()
    if u == 0:
      iou = 1.0
    else:
      iou = i / u
    ious.append(iou)
  miou = np.mean(ious)
  return miou


def all_subclasses(cls):
  return set(cls.__subclasses__()).union([s for c in cls.__subclasses__() for s in all_subclasses(c)])


def get_lr_schedulers(optimiser, args, last_epoch=-1):
  last_epoch = -1 if last_epoch ==0 else last_epoch
  lr_schedulers = []
  if args.lr_schedulers is None:
    return lr_schedulers
  if 'exponential' in args.lr_schedulers:
    lr_schedulers += [torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=args.lr_decay, last_epoch=last_epoch)]
  if 'step' in args.lr_schedulers:
    lr_schedulers += [torch.optim.lr_scheduler.MultiStepLR(optimiser, milestones=[15, 20],
                                                          last_epoch=last_epoch)]
  return lr_schedulers
