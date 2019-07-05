import os
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image
from utils.Constants import font
from utils.util import ToLabel


def load_weights(model, optimizer, loadepoch, model_dir):
    start_epoch = 0
    # load saved model if specified
    if loadepoch is not None:
      print('Loading checkpoint {}@Epoch {}{}...'.format(font.BOLD, loadepoch, font.END))
      if loadepoch == '0':
        # transform, checkpoint provided by RGMP
        load_name = 'weights.pth'
        state = model.state_dict()
        checkpoint = torch.load(load_name)
        checkpoint = {"model": OrderedDict([(k.replace("module.", ""), v) for k, v in checkpoint.items()])}
        checkpoint['epoch'] = 0
      else:
        load_name = os.path.join(model_dir,
                                 '{}.pth'.format(loadepoch))
        state = model.state_dict()
        checkpoint = torch.load(load_name)
        start_epoch = checkpoint['epoch'] + 1

      checkpoint_valid = {k: v for k, v in checkpoint['model'].items() if k in state and state[k].shape == v.shape}
      missing_keys = np.setdiff1d(list(state.keys()),list(checkpoint_valid.keys()))
      for key in missing_keys:
        checkpoint_valid[key] = state[key]

      model.load_state_dict(checkpoint_valid)
      if 'optimizer' in checkpoint.keys():
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
      if 'optimizer_extra' in checkpoint.keys():
        optimizer.load_state_dict(checkpoint['optimizer_extra'])
      if 'pooling_mode' in checkpoint.keys():
        POOLING_MODE = checkpoint['pooling_mode']
      del checkpoint
      torch.cuda.empty_cache()
      print('Loaded weights from {}'.format(load_name))

    return model, optimizer, start_epoch


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


def save_checkpoint(epoch, iou_mean, model, optimiser, save_name):
  torch.save({'epoch': epoch,
              'model': model.state_dict(),
              'optimizer': optimiser.state_dict(),
              },
             save_name)
  print("Saving epoch {} with IOU {}".format(epoch, iou_mean))