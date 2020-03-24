import inspect

import numpy as np
import torch
from torch.distributed import all_reduce
from network.EmbeddingNetwork import Resnet3dSpatialEmbedding
from network.models import BaseNetwork


def ToOneHot(labels, num_objects):
  print(labels)
  labels = labels.view(-1, 1)
  labels = torch.eye(num_objects).index_select(dim=0, index=labels)
  return labels.cuda()


def ToLabel(E):
  fgs = np.argmax(E, axis=1).astype(np.float32)
  return fgs.astype(np.uint8)


def get_iou(gt, pred):
  i = np.logical_and(pred > 0, gt > 0).sum()
  u = np.logical_or(pred > 0, gt > 0).sum()
  if u == 0:
    iou = 1.0
  else:
    iou = i / u
  return iou


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


def iou_fixed_torch(pred, gt, exclude_last=False):
  pred = torch.argmax(pred, dim=1).int()
  ious = []
  num_frames = pred.shape[0]
  end = num_frames
  if exclude_last:
    end -= 1
  for t in range(0, end):
    i = ((pred[t] > 0) * (gt[t] > 0)).float().sum()
    u = ((pred[t] + gt[t]) > 0).float().sum()
    if u == 0:
      iou = torch.cuda.FloatTensor([1.0]).sum()
    else:
      iou = i.float() / u.float()
    ious.append(iou.float())
  miou = torch.stack(ious).float().mean()
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


def show_image_summary(count, foo, input_var, masks_guidance, target, pred):
  for index in range(input_var.shape[2]):
    foo.add_images("data/input" + str(index), input_var[:, :3, index], count)
    if masks_guidance is not None:
      tensor = masks_guidance[:, :, index] if len(masks_guidance.shape) > 4 else masks_guidance
      foo.add_images("data/guidance" + str(index), tensor.repeat(1, 3, 1, 1), count)
  # foo.add_image("data/loss_image", loss_image.unsqueeze(1), count)
  if len(target.shape) < 5:
    target = target.unsqueeze(2)
    pred = pred.unsqueeze(2)
  for index in range(target.shape[2]):
    foo.add_images("data/target"+ str(index), target[:, :, index].repeat(1,3,1,1), count)
    foo.add_images("data/pred"+ str(index), torch.argmax(pred, dim=1)[:, index].unsqueeze(1).repeat(1,3,1,1), count)


def get_model(args, network_models):
  model_classes = all_subclasses(BaseNetwork)
  class_index = [cls.__name__ for cls in model_classes].index(network_models[args.network])
  model_class = list(model_classes)[class_index]
  spec = inspect.signature(model_class.__init__)
  fn_args = spec._parameters
  params = {}
  if 'n_classes' in fn_args:
    params['n_classes'] = args.n_classes
  if 'tw' in fn_args:
    params['tw'] = args.tw
  if 'e_dim' in fn_args:
    params['e_dim'] = args.embedding_dim
  model = model_class(**params)
  return model


def init_torch_distributed():
  print("devices available: {}".format(torch.cuda.device_count()))
  torch.distributed.init_process_group(
    'nccl',
    init_method='env://',
  )


def cleanup_env():
  """
  Destroy the default process group.

  :return:
  """
  torch.distributed.destroy_process_group()

def reduce_tensor(tensor, args):
  from apex.parallel import ReduceOp
  rt = tensor.clone()
  all_reduce(rt, op=ReduceOp.SUM)
  rt /= args.world_size
  return rt
