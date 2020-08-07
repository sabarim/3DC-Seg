from abc import abstractmethod
import numpy as np
import torch
from torch import nn

from utils import Constants


class BaseNetwork(nn.Module):
  def __init__(self, tw, pixel_mean, pixel_std):
    super(BaseNetwork, self).__init__()
    self.tw = tw
    self.register_buffer('mean', torch.FloatTensor(pixel_mean).view(1, 3, 1, 1, 1))
    self.register_buffer('std', torch.FloatTensor(pixel_std).view(1, 3, 1, 1, 1))

  @abstractmethod
  def load_pretrained(self, wts):
    print('Loading pretrained weights from {} {}{}...'.format(Constants.font.BOLD, wts, Constants.font.END))
    chkpt = torch.load(wts, map_location=torch.device('cpu'))
    state = self.state_dict()
    checkpoint_valid = {k: v for k, v in chkpt.items() if k in state and state[k].shape == v.shape}
    missing_keys = np.setdiff1d(list(state.keys()), list(checkpoint_valid.keys()))

    if len(missing_keys) > 0:
      print(missing_keys)
      print("WARN: {} / {}keys are found missing in the loaded model weights.".format(len(missing_keys),
                                                                                      len(state.keys())))
    for key in missing_keys:
      checkpoint_valid[key] = state[key]

    self.load_state_dict(checkpoint_valid)
