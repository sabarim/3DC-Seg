import inspect
import os

import numpy as np
import torch
import torch.distributed as dist
from deprecated import deprecated
from torch import nn
from torch.distributed import all_reduce

from datasets.BaseDataset import BaseDataset
from network.models import BaseNetwork
from utils.Constants import ADAM_OPTIMISER, PRED_LOGITS, PRED_EMBEDDING, PRED_SEM_SEG


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


def get_lr_schedulers(optimiser, cfg, last_epoch=-1):
  last_epoch = -1 if last_epoch ==0 else last_epoch
  all_schedulers = inspect.getmembers(torch.optim.lr_scheduler)
  lr_schedulers = []

  if 'exponential' in cfg.SOLVER.LR_SCHEDULERS:
    lr_schedulers += [torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=cfg.SOLVER.GAMMA, last_epoch=last_epoch)]
  if 'step' in cfg.SOLVER.LR_SCHEDULERS:
    lr_schedulers += [torch.optim.lr_scheduler.MultiStepLR(optimiser, milestones=cfg.SOLVER.STEPS,
                                                           last_epoch=last_epoch)]
  return lr_schedulers

@deprecated
def get_lr_schedulers_args(optimiser, args, last_epoch=-1):
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


def show_image_summary(count, foo, in_dict, target_dict, pred_dict):
  # show inputs
  for k, v in in_dict.items():
    if len(v.shape) < 5:
      v = v.unsqueeze(2)
    for index in range(v.shape[2]):
      foo.add_images("data/{}_{}".format(k, index), v[:, :3, index], count)

  # show targets
  for k, v in target_dict.items():
    if len(v.shape) < 5:
      target = v.unsqueeze(2)
      for index in range(target.shape[2]):
        foo.add_images("data/{}_{}".format(k, str(index)), target[:, :, index].repeat(1, 3, 1, 1), count)

    # show predictions
  for k, v in pred_dict.items():
    pred = v.unsqueeze(2)
    for index in range(pred.shape[2]):
      foo.add_images("pred/{}_{}".format(k, str(index)), pred[:, index].unsqueeze(1).repeat(1,3,1,1), count)


def get_model(cfg):
  model_classes = all_subclasses(BaseNetwork)
  class_index = [cls.__name__ for cls in model_classes].index(cfg.MODEL.NETWORK)
  model_class = list(model_classes)[class_index]
  model = model_class(cfg)
  # if cfg.MODEL.PRETRAINED:
  #   model.load_pretrained(cfg.MODEL.WEIGHTS)
  return model


def get_model_from_args(args, network_models):
  model_classes = all_subclasses(BaseNetwork)
  modules = all_subclasses(nn.Module)
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
  if 'inter_block' in fn_args:
    class_index = [cls.__name__ for cls in modules].index(args.inter_block)
    module_class = list(modules)[class_index]
    params['inter_block'] = module_class
  if 'refine_block' in fn_args:
    class_index = [cls.__name__ for cls in modules].index(args.refine_block)
    module_class = list(modules)[class_index]
    params['refine_block'] = module_class

  model = model_class(**params)
  return model


def get_datasets(cfg):
  dataset_classes = all_subclasses(BaseDataset)
  try:
    class_index = [cls.__name__ for cls in dataset_classes].index(cfg.DATASETS.TRAIN)
  except:
    raise ValueError("Dataset {} not found.".format(cfg.DATASETS.TRAIN))

  train_dataset_class = list(dataset_classes)[class_index]
  train_dataset = build_dataset(train_dataset_class, True, cfg)

  try:
    class_index = [cls.__name__ for cls in dataset_classes].index(cfg.DATASETS.TEST)
  except:
    raise ValueError("Dataset {} not found.".format(cfg.DATASETS.TEST))
  test_dataset_class = list(dataset_classes)[class_index]
  test_dataset = build_dataset(test_dataset_class, False, cfg)

  return train_dataset, test_dataset


def build_dataset(_class, is_train, cfg):
  spec = inspect.signature(_class.__init__)
  fn_args = spec._parameters
  params = {}
  params['root'] = cfg.DATASETS.TRAIN_ROOT if is_train else cfg.DATASETS.TEST_ROOT
  params['mode'] = 'train' if is_train else "test"
  params['resize_mode'] = cfg.INPUT.RESIZE_MODE_TRAIN if is_train else cfg.INPUT.RESIZE_MODE_TEST
  params['resize_shape'] = cfg.INPUT.RESIZE_SHAPE_TRAIN if is_train else cfg.INPUT.RESIZE_SHAPE_TEST

  # cfg_params = dict(cfg.items())['DATASETS']
  cfg_params = dict(list(dict(cfg.items())['INPUT'].items()) + list(dict(cfg.items())['DATASETS'].items()))
  missing_params = []
  for p in fn_args:
    if p in params:
      continue

    if p.upper() in cfg_params:
      params[str(p)] = cfg_params[p.upper()]
    else:
      missing_params += [p]

  print("Dataset parameters {} are missing in the config file.".format(missing_params))
  # params['random_instance'] = cfg.DATASETS.RANDOM_INSTANCE
  dataset = _class(**params)

  return dataset


def get_optimiser(model, cfg):
  if cfg.TRAINING.OPTIMISER == ADAM_OPTIMISER:
    opt = torch.optim.Adam(model.parameters(), lr=cfg.TRAINING.BASE_LR)
  else:
    raise ValueError("Unknown optimiser {}".format(cfg.TRAINING.OPTIMISER))

  return opt


def _find_free_port():
  import socket
  port_range = list(range(1230, 1250))
  port_range += list(range(8085, 8099))
  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  for port in port_range:
    try:
      sock.bind(('', port))
      sock.close()
      return port
    except OSError:
      continue
  # NOTE: there is still a chance the port could be taken by other processes.
  return port


def init_torch_distributed(port):
  print("devices available: {}".format(torch.cuda.device_count()))
  #port = _find_free_port()
  print("Using port {} for torch distributed.".format(port))
  if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    torch.distributed.init_process_group(
      'nccl',
      init_method='env://',
    )
  else:
    dist_url = "tcp://127.0.0.1:{}".format(port)
    try:
      dist.init_process_group(
        backend="NCCL",
        init_method=dist_url, world_size=1, rank=0
      )
    except Exception as e:
      print("Process group URL: {}".format(dist_url))
      raise e


def get_rank():
  if not dist.is_available():
    return 0
  if not dist.is_initialized():
    return 0
  return dist.get_rank()


def is_main_process():
  return get_rank() == 0


def synchronize():
  """
  Helper function to synchronize (barrier) among all processes when
  using distributed training
  """
  if not dist.is_available():
    return
  if not dist.is_initialized():
    return
  world_size = dist.get_world_size()
  if world_size == 1:
    return
  dist.barrier()


def cleanup_env():
  """
  Destroy the default process group.

  :return:
  """
  print("Destroying distributed processes.")
  torch.distributed.destroy_process_group()

def reduce_tensor(tensor, world_size):
  from apex.parallel import ReduceOp
  rt = tensor.clone()
  all_reduce(rt, op=ReduceOp.SUM)
  rt /= world_size
  return rt


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