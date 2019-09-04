import os

import numpy as np
from PIL import Image

from datasets.DAVIS import DAVIS, DAVISEval
from utils.Resize import ResizeMode, resize


class DAVIS16(DAVIS):
  def __init__(self, root, imset='2017/train.txt', resolution='480p', is_train=False,
               random_instance=False, num_classes=2, crop_size=None,temporal_window=5, min_temporal_gap=2,
               max_temporal_gap=8, resize_mode=ResizeMode.FIXED_SIZE, proposal_dir=None, augmentors=None):
    super(DAVIS16, self).__init__(root, imset, is_train=is_train, crop_size=crop_size,
                                  temporal_window=temporal_window, random_instance=random_instance,
                                  resize_mode=resize_mode, proposal_dir=proposal_dir,
                                  max_temporal_gap=temporal_window *2, min_temporal_gap=1)

  def __getitem__(self, item):
    input_dict = super(DAVIS16, self).__getitem__(item)
    del input_dict['masks_guidance']
    input_dict['target'] = input_dict['raw_masks']
    return input_dict

  def read_frame(self, shape, video, f, instance_id=None):
    # use a blend of both full random instance as well as the full object
    img_file = os.path.join(self.image_dir, video, '{:05d}.jpg'.format(f))
    raw_frames = np.array(Image.open(img_file).convert('RGB')) / 255.
    try:
      mask_file = os.path.join(self.mask_dir, video, '{:05d}.png'.format(f))  # allways return first frame mask
      raw_mask = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
    except:
      mask_file = os.path.join(self.mask_dir, video, '00000.png')
      raw_mask = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)

    raw_mask = (raw_mask != 0).astype(np.uint8)
    tensors_resized = resize({"image":raw_frames, "mask":raw_mask, "proposals": raw_mask},
                             self.resize_mode, shape)

    return tensors_resized["image"] / 255.0, tensors_resized["mask"], tensors_resized["proposals"], \
           tensors_resized["proposals"]

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


class DAVIS16PredictOne(DAVIS16):
  def __init__(self, root, imset='2017/train.txt', resolution='480p', is_train=False,
               random_instance=False, num_classes=2, crop_size=None,temporal_window=5, min_temporal_gap=2,
               max_temporal_gap=8, resize_mode=ResizeMode.FIXED_SIZE, proposal_dir=None, augmentors=None,
               predict_centre = False):
    super(DAVIS16PredictOne, self).__init__(root, imset, is_train=is_train, crop_size=crop_size,
                                    temporal_window=temporal_window, random_instance=random_instance,
                                    resize_mode=resize_mode, proposal_dir=proposal_dir)
    self.predict_centre = predict_centre

  def __getitem__(self, item):
    input_dict = super(DAVIS16PredictOne, self).__getitem__(item)
    if self.predict_centre:
      assert self.temporal_window % 2 != 0
      centre = int(np.floor(self.temporal_window / 2))
      input_dict['target'] = input_dict['raw_masks'][:, centre]
    else:
      input_dict['target'] = input_dict['raw_masks'][:,-1]
    return input_dict


class DAVIS16PredictOneEval(DAVIS16Eval):
  def __init__(self, root, imset='2016/val.txt', resolution='480p', is_train=False,
               random_instance=False, num_classes=2, crop_size=None,temporal_window=5, min_temporal_gap=2,
               max_temporal_gap=8, resize_mode=ResizeMode.FIXED_SIZE, proposal_dir=None, augmentors=None,
               predict_centre = False):
    super(DAVIS16PredictOneEval, self).__init__(root, imset, is_train=is_train, crop_size=crop_size,
                                    temporal_window=temporal_window, random_instance=random_instance,
                                    resize_mode=resize_mode, proposal_dir=proposal_dir)
    self.predict_centre = predict_centre

  def __getitem__(self, item):
    input_dict = super(DAVIS16PredictOneEval, self).__getitem__(item)
    if self.predict_centre:
      assert self.temporal_window % 2 != 0
      centre = int(np.floor(self.temporal_window / 2))
      input_dict['target'] = input_dict['raw_masks'][:, centre]
    else:
      input_dict['target'] = input_dict['raw_masks'][:,-1]
    return input_dict