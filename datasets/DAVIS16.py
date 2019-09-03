import numpy as np
from datasets.DAVIS import DAVIS, DAVISEval
from utils.Resize import ResizeMode


class DAVIS16(DAVIS):
  def __init__(self, root, imset='2016/train.txt', resolution='480p', is_train=False,
               random_instance=False, num_classes=2, crop_size=None,temporal_window=5, min_temporal_gap=2,
               max_temporal_gap=8, resize_mode=ResizeMode.FIXED_SIZE, proposal_dir=None, augmentors=None):
    super(DAVIS16, self).__init__(root, imset, is_train=is_train, crop_size=crop_size,
                                    temporal_window=temporal_window, random_instance=random_instance,
                                    resize_mode=resize_mode, proposal_dir=proposal_dir)

  def __getitem__(self, item):
    input_dict = super(DAVIS16, self).__getitem__(item)
    del input_dict['masks_guidance']
    input_dict['target'] = input_dict['raw_masks']
    return input_dict

  def get_support_indices(self, index, sequence):
    if index == 0:
      support_indices = np.repeat([0], self.temporal_window)
    else:
      support_indices = np.arange(max(0, (index - self.temporal_window) + 1), index + 1)
      support_indices = np.append(support_indices, np.repeat([index], self.temporal_window - len(support_indices)))

    # print(support_indices)
    return support_indices


class DAVIS16Eval(DAVISEval):
  def __init__(self, root, imset='2016/val.txt', resolution='480p', is_train=False,
               random_instance=False, num_classes=2, crop_size=None, temporal_window=5, min_temporal_gap=2,
               max_temporal_gap=8, resize_mode=ResizeMode.FIXED_SIZE, proposal_dir=None, augmentors=None):
    super(DAVIS16Eval, self).__init__(root, imset, is_train=is_train, crop_size=crop_size,
                                  temporal_window=temporal_window, random_instance=random_instance,
                                  resize_mode=resize_mode, proposal_dir=proposal_dir)

  def __getitem__(self, item):
    input_dict = super(DAVIS16Eval, self).__getitem__(item)
    del input_dict['masks_guidance']
    input_dict['target'] = input_dict['raw_masks']
    return input_dict


class DAVIS16PredictCentre(DAVIS16):
  def __init__(self, root, imset='2016/train.txt', resolution='480p', is_train=False,
               random_instance=False, num_classes=2, crop_size=None,temporal_window=5, min_temporal_gap=2,
               max_temporal_gap=8, resize_mode=ResizeMode.FIXED_SIZE, proposal_dir=None, augmentors=None):
    super(DAVIS16PredictCentre, self).__init__(root, imset, is_train=is_train, crop_size=crop_size,
                                    temporal_window=temporal_window, random_instance=random_instance,
                                    resize_mode=resize_mode, proposal_dir=proposal_dir)

  def __getitem__(self, item):
    input_dict = super(DAVIS16, self).__getitem__(item)
    input_dict['target'] = input_dict['raw_masks'][-1]
    return input_dict