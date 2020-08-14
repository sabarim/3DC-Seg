import glob
import os
from collections import OrderedDict
import numpy as np
import torch
from PIL import Image
from deprecated import deprecated

import utils.Constants as Constants
from utils.util import ToLabel


def load_weightsV2(model, optimiser, wts_file, model_dir):
  start_epoch = 0
  start_iter = 0
  state = model.state_dict()
  checkpoint = None
  # load saved model if specified
  if wts_file is None:
    # load checkpoint file if it exists -- > file  format checkpoint_<iteration>
    chkpts = glob.glob(model_dir + "/*.pth")
    chkpts = [c for c in chkpts if "checkpoint_" in c]
    if len(chkpts) > 0:
      chkpts.sort()
      load_name = chkpts[-1]
      print('Loading checkpoint {}@Epoch {}{}...'.format(Constants.font.BOLD, load_name, Constants.font.END))
      checkpoint = torch.load(load_name)
      start_epoch = checkpoint['epoch'] + 1
      start_iter = checkpoint['iter'] + 1
  else:
    checkpoint = torch.load(wts_file)
    start_epoch = checkpoint['epoch'] + 1 if 'epoch' in checkpoint else 0
    start_iter = checkpoint['iter'] + 1 if 'iter' in checkpoint else 0
    load_name = wts_file

  if checkpoint is not None:
    checkpoint['model'] = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
    checkpoint_valid = {k: v for k, v in checkpoint['model'].items() if k in state and state[k].shape == v.shape}
    missing_keys = np.setdiff1d(list(state.keys()), list(checkpoint_valid.keys()))

    if len(missing_keys) > 0:
      print(missing_keys)
      print("WARN: {} / {}keys are found missing in the loaded model weights.".format(len(missing_keys),
                                                                                      len(state.keys())))
    for key in missing_keys:
      checkpoint_valid[key] = state[key]

    model.load_state_dict(checkpoint_valid)
    if 'optimizer' in checkpoint.keys():
      try:
        optimiser.load_state_dict(checkpoint['optimizer'])
      except:
        print("WARN: Optimiser state could not be resumed. Resetting start epoch to 0....")
        start_epoch = 0
        start_iter = 0

    del checkpoint
    torch.cuda.empty_cache()
    print('Loaded weights from {}'.format(load_name))

  return model, optimiser, start_epoch, start_iter


@deprecated
def load_weights(model, optimizer, args, model_dir, scheduler, amp = None):
    start_epoch = 0
    best_iou_train = 0
    best_iou_eval = 0
    best_loss_train = 0
    best_loss_eval = 0
    loadepoch = args.loadepoch
    state = model.state_dict()
    # load saved model if specified
    if loadepoch is not None:
      print('Loading checkpoint {}@Epoch {}{}...'.format(Constants.font.BOLD, loadepoch, Constants.font.END))
      # load checkpoint file if it exists -- > file  format checkpoint_<iteration>
      chkpts = glob.glob(model_dir + "/*.pth")
      chkpts = [c for c in chkpts if "checkpoint_" in c]
      if len(chkpts) > 0:
          chkpts.sort()
          load_name = chkpts[-1]
          checkpoint = torch.load(load_name)
          start_epoch = checkpoint['epoch'] + 1
      elif 'pretrain' in loadepoch:
        load_name = os.path.join(Constants.MODEL_ROOT, loadepoch + '.pth')
        checkpoint = load_pretrained_weights(load_name)
      elif loadepoch.startswith("resnet"):
        # transform, checkpoint provided by RGMP
        load_name = os.path.join("saved_models", '{}.pth'.format(loadepoch))
        state = model.state_dict()
        checkpoint = torch.load(load_name)['state_dict']
        start_epoch = 0
        checkpoint = {"model": OrderedDict([('encoder.' + k.lower(), v)
                                            for k, v in checkpoint.items()]), "epoch": start_epoch}
      elif loadepoch == 'siam':
        # transform, checkpoint provided by RGMP
        load_name2d = 'saved_models/rgmp.pth'
        load_name3d = 'saved_models/resnet-50-kinetics.pth'
        load_name = {1: 'saved_models/youtubevos_pretrain.pth', 2: 'saved_models/rgmp.pth'}
        checkpoint2d = torch.load(load_name2d)
        checkpoint = load_pretrained_weights(load_name3d)
        encoder2d_dict = OrderedDict([(k.lower().replace('module.encoder', 'encoder2d'), v)
                                            for k, v in checkpoint2d.items() if "encoder" in k.lower()])
        #FIXME: uncomment for using rgmp pretrained weights
        #checkpoint['model'].update(encoder2d_dict)
        # checkpoint = {"model": OrderedDict([(k.replace("module.", ""), v) for k, v in checkpoint.items()])}
      elif loadepoch == 'kinetics':
        # transform, checkpoint provided by RGMP
        load_name = Constants.MODEL_ROOT + 'resnet-50-kinetics.pth'
        state = model.state_dict()
        checkpoint = torch.load(load_name)
        # checkpoint = {"model": OrderedDict([(k.replace("module.", ""), v) for k, v in checkpoint.items()])}
        checkpoint = {"model": OrderedDict([(k.lower().replace('module.', 'encoder.'), v)
                                            for k, v in checkpoint['state_dict'].items()]), 'epoch': 0}
      elif ("csn/" in loadepoch or "2+1d" in loadepoch) and "model_best" not in loadepoch:
        load_name = os.path.join('saved_models/',
                                 '{}.pth'.format(loadepoch))
        checkpoint = torch.load(load_name)
        start_epoch = 0
        checkpoint = {"model" : OrderedDict([('encoder.' + k.lower(), v)
                                            for k, v in checkpoint.items()]), "epoch" : start_epoch}
      elif len(loadepoch.split("/")) > 1:
        load_name = os.path.join(Constants.MODEL_ROOT, loadepoch + '.pth')
        checkpoint = load_pretrained_weights(load_name)
      else:
        load_name = os.path.join('saved_models/', args.network_name,
                     '{}.pth'.format(loadepoch))
        checkpoint = torch.load(load_name)
        start_epoch = checkpoint['epoch'] + 1 if (args.task == "train" and "epoch" in checkpoint) else 0
        # checkpoint["model"] = OrderedDict([(k.replace("module.", ""), v) for k, v in checkpoint["model"].items()])
      # remove module. prefix
      checkpoint['model'] = {k.replace('module.', ''):v for k, v in checkpoint['model'].items()}
      checkpoint_valid = {k: v for k, v in checkpoint['model'].items() if k in state and state[k].shape == v.shape}
      missing_keys = np.setdiff1d(list(state.keys()),list(checkpoint_valid.keys()))
      
      if len(missing_keys) > 0:
        print(missing_keys)
        print("WARN: {} / {}keys are found missing in the loaded model weights.".format(len(missing_keys),
                                                                                        len(state.keys())))
      for key in missing_keys:
        checkpoint_valid[key] = state[key]
      
      model.load_state_dict(checkpoint_valid)
      if args.task == 'train':
        if 'optimizer' in checkpoint.keys() :
          optimizer.load_state_dict(checkpoint['optimizer'])
          lr = optimizer.param_groups[0]['lr']
        if 'scheduler' in checkpoint.keys() and checkpoint['scheduler'] is not None and scheduler is not None:
          scheduler.load_state_dict(checkpoint['scheduler'].state_dict())
        if 'amp' in checkpoint.keys() and checkpoint['amp'] is not None and amp is not None:
          amp = checkpoint['amp']
        if 'best_iou' in checkpoint.keys() and checkpoint['task'] == 'train':
          best_iou_train = checkpoint['best_iou']
          best_loss_train = checkpoint['loss']
        elif 'best_iou' in checkpoint.keys():
          best_iou_eval = checkpoint['best_iou']
          best_loss_eval = checkpoint['loss']

      del checkpoint
      torch.cuda.empty_cache()
      print('Loaded weights from {}'.format(load_name))

    return model, optimizer, start_epoch, best_iou_train, best_iou_eval, best_loss_train, best_loss_eval, amp


def load_pretrained_weights(load_name):
  checkpoint = torch.load(load_name)
  checkpoint['epoch'] = 0
  keys = ['optimizer', 'scheduler', 'best_iou']
  for key in keys:
    del checkpoint[key]
  return checkpoint


def save_results(all_E, info, num_frames, path, palette):
  if not os.path.exists(path):
    os .makedirs(path)
  for f in range(num_frames):
    E = all_E[0, :, f].numpy()
    # make hard label
    E = ToLabel(E)

    (lh, uh), (lw, uw) = info['pad']
    E = E[lh[0]:-uh[0], lw[0]:-uw[0]]

    img_E = Image.fromarray(E)
    img_E.putpalette(palette)
    img_E.save(os.path.join(path, '{:05d}.png'.format(f)))


def save_checkpointV2(epoch, iter, model, optimiser, save_name):
  torch.save({'epoch': epoch,
              'model': model.state_dict(),
              'optimizer': optimiser.state_dict(),
              'iter': iter
              },
             save_name)
  print("Saving epoch {}".format(epoch))


def save_checkpoint(epoch, iou_mean, loss_mean, model, optimiser, save_name, is_train, scheduler, amp = None):
  torch.save({'epoch': epoch,
              'model': model.state_dict(),
              'optimizer': optimiser.state_dict(),
              'best_iou': iou_mean,
              'loss': loss_mean,
              'task': 'train' if is_train else 'eval',
              'scheduler': scheduler,
              'amp': amp.state_dict() if amp is not None else amp
              },
             save_name)
  print("Saving epoch {} with IOU {}".format(epoch, iou_mean))
