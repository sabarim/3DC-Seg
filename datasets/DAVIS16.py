import os
import random
import glob
import numpy as np
from PIL import Image

from datasets.DAVIS import DAVIS, DAVISEval
from util import get_one_hot_vectors
from utils.Resize import ResizeMode, resize


class DAVIS16(DAVIS):
  def __init__(self, root, imset='2017/train.txt', resolution='480p', is_train=False,
               random_instance=False, num_classes=2, crop_size=None,temporal_window=8, min_temporal_gap=2,
               max_temporal_gap=8, resize_mode=ResizeMode.FIXED_SIZE, proposal_dir=None, augmentors=None):
    super(DAVIS16, self).__init__(root, imset, is_train=is_train, crop_size=crop_size,
                                  temporal_window=temporal_window, random_instance=random_instance,
                                  resize_mode=resize_mode, proposal_dir=proposal_dir,
                                  max_temporal_gap=max_temporal_gap, min_temporal_gap=1)
    self.random_instance_ids = {}
    self.num_classes = num_classes

  def __getitem__(self, item):
    input_dict = super(DAVIS16, self).__getitem__(item)
    del input_dict['masks_guidance']
    input_dict['target'] = input_dict['raw_masks']
    return input_dict

  def set_video_id(self, video):
    self.current_video = video
    self.start_index = self.get_start_index(video)
    self.img_list = list(glob.glob(os.path.join(self.image_dir, self.current_video, '*.jpg')))
    self.img_list.sort()
    instance_ids = list(range(self.num_objects[video] + 1))
    instance_ids.remove(0)
    self.random_instance_ids[video] = random.choice(instance_ids)

  def get_video_ids(self):
    # shuffle the list for training
    return random.sample(self.videos, len(self.videos)) if self.is_train else self.videos

  def __len__(self):
    if self.is_train or self.current_video is None:
      return super(DAVIS16, self).__len__()
    else:
      return self.num_frames[self.current_video]

  def read_frame(self, shape, video, f, instance_id=None, support_indices=None):
    # use a blend of both full random instance as well as the full object
    img_file = os.path.join(self.image_dir, video, '{:05d}.jpg'.format(f))
    raw_frames = np.array(Image.open(img_file).convert('RGB')) / 255.
    try:
      mask_file = os.path.join(self.mask_dir, video, '{:05d}.png'.format(f))  # allways return first frame mask
      raw_mask = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
    except:
      mask_file = os.path.join(self.mask_dir, video, '00000.png')
      raw_mask = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)

    mask_void = (raw_mask == 255).astype(np.uint8)
    raw_mask[raw_mask == 255] = 0
    if self.num_classes == 2:
      raw_mask = (raw_mask != 0).astype(np.uint8) if not self.random_instance else \
        (raw_mask == instance_id).astype(np.uint8)
    # if f==min(support_indices) and self.random_instance:
    #   tensors_resized = resize({"image": raw_frames, "mask": raw_mask, "proposals": raw_mask},
    #                            ResizeMode.BBOX_CROP_AND_RESIZE_FIXED_SIZE, shape)
    # else:
    tensors_resized = resize({"image":raw_frames, "mask":raw_mask, "proposals": raw_mask},
                             self.resize_mode, shape)

    return tensors_resized["image"] / 255.0, tensors_resized["mask"], tensors_resized["proposals"], \
           tensors_resized["proposals"], mask_void

  def get_support_indices(self, index, sequence):
    # index should be start index of the clip
    if self.is_train:
      index_range = np.arange(index, min(self.num_frames[sequence],
                                         (index + max(self.max_temporal_gap, self.temporal_window))))
    else:
      index_range = np.arange(index,
                              min(self.num_frames[sequence], (index + self.temporal_window)))

    support_indices = np.random.choice(index_range, min(self.temporal_window, len(index_range)), replace=False)
    support_indices = np.sort(np.append(support_indices, np.repeat([index],
                                                                   self.temporal_window - len(support_indices))))

    # print(support_indices)
    return support_indices


class DAVIS16Eval(DAVISEval):
  def __init__(self, root, imset='2016/val.txt', resolution='480p', is_train=False,
               random_instance=False, num_classes=2, crop_size=None, temporal_window=5, min_temporal_gap=2,
               max_temporal_gap=8, resize_mode=ResizeMode.FIXED_SIZE, proposal_dir=None, augmentors=None):
    super(DAVIS16Eval, self).__init__(root, imset, is_train=is_train, crop_size=crop_size,
                                  temporal_window=temporal_window, random_instance=random_instance,
                                  resize_mode=resize_mode, proposal_dir=proposal_dir)

  def get_support_indices(self, index, sequence):
    # index should be start index of the clip
    if self.is_train:
      index_range = np.arange(index, min(self.num_frames[sequence],
                                         (index + max(self.max_temporal_gap, self.temporal_window) + 1)))
    else:
      index_range = np.arange(index, min(self.num_frames[sequence], (index + self.temporal_window)))

    support_indices = np.random.choice(index_range, min(self.temporal_window, len(index_range)), replace=False)
    support_indices = np.sort(np.append(support_indices, np.repeat([index],
                                                                   self.temporal_window - len(support_indices))))

    # print(support_indices)
    return support_indices

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


class DAVIS17MaskGuidance(DAVIS16):
  def __init__(self, root, imset='2017/train.txt', resolution='480p', is_train=False,
               random_instance=False, num_classes=2, crop_size=None,temporal_window=8, min_temporal_gap=2,
               max_temporal_gap=8, resize_mode=ResizeMode.FIXED_SIZE, proposal_dir=None, augmentors=None):
    super(DAVIS17MaskGuidance, self).__init__(root, imset, is_train=is_train, crop_size=crop_size,
                                              temporal_window=temporal_window, random_instance=random_instance,
                                              resize_mode=resize_mode, proposal_dir=proposal_dir,
                                  max_temporal_gap=max_temporal_gap, min_temporal_gap=1)

  def __getitem__(self, item):
    input_dict = super(DAVIS17MaskGuidance, self).__getitem__(item)
    input_dict['masks_guidance'] = input_dict['raw_masks'][:, 0]
    return input_dict

  def read_frame(self, shape, video, f, instance_id=None, support_indices=None):
    # use a blend of both full random instance as well as the full object
    img_file = os.path.join(self.image_dir, video, '{:05d}.jpg'.format(f))
    raw_frames = np.array(Image.open(img_file).convert('RGB')) / 255.
    try:
      mask_file = os.path.join(self.mask_dir, video, '{:05d}.png'.format(f))  # allways return first frame mask
      raw_mask = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
    except:
      mask_file = os.path.join(self.mask_dir, video, '00000.png')
      raw_mask = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)

    
    mask_void = (raw_mask == 255).astype(np.uint8)
    raw_mask[raw_mask == 255] = 0
    raw_mask = (raw_mask != 0).astype(np.uint8) if not self.random_instance else \
      (raw_mask == instance_id).astype(np.uint8)
    tensors_resized = resize({"image":raw_frames, "mask":raw_mask, "proposals": raw_mask},
                           self.resize_mode, shape)

    return tensors_resized["image"] / 255.0, tensors_resized["mask"], tensors_resized["proposals"], \
           tensors_resized["proposals"], mask_void


class DAVISSiam3d(DAVIS16):
  def __init__(self, root, resolution='480p', is_train=False,
               random_instance=False, num_classes=2, crop_size=None,temporal_window=8, min_temporal_gap=2,
               max_temporal_gap=8, resize_mode=ResizeMode.FIXED_SIZE, proposal_dir=None, augmentors=None):
    imset = "2017/train.txt" if is_train else "2017/val.txt"
    super(DAVISSiam3d, self).__init__(root, imset, is_train=is_train, crop_size=crop_size,
                                  temporal_window=temporal_window, random_instance=random_instance,
                                  resize_mode=resize_mode, proposal_dir=proposal_dir,
                                  max_temporal_gap=max_temporal_gap, min_temporal_gap=1)

  def __getitem__(self, item):
    input_dict = super(DAVIS16, self).__getitem__(item)
    input_dict['masks_guidance'] = input_dict['raw_masks'][:, 0]
    input_dict['target'] = input_dict['raw_masks'][:, 1:]
    return input_dict

  def read_frame(self, shape, video, f, instance_id=None, support_indices=None):
    # use a blend of both full random instance as well as the full object
    img_file = os.path.join(self.image_dir, video, '{:05d}.jpg'.format(f))
    raw_frames = np.array(Image.open(img_file).convert('RGB')) / 255.
    try:
      mask_file = os.path.join(self.mask_dir, video, '{:05d}.png'.format(f))  # allways return first frame mask
      raw_mask = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
    except:
      mask_file = os.path.join(self.mask_dir, video, '00000.png')
      raw_mask = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)

    mask_void = (raw_mask == 255).astype(np.uint8)
    raw_mask[raw_mask == 255] = 0
    # pick a random instance for each clip during training while pick a common instance for the video during evaluation
    instance_id = instance_id if self.is_train else self.random_instance_ids[self.current_video]
    raw_mask = (raw_mask != 0).astype(np.uint8) if not self.random_instance else \
      (raw_mask == instance_id).astype(np.uint8)
    tensors_resized = resize({"image":raw_frames, "mask":raw_mask, "proposals": raw_mask},
                           self.resize_mode, shape)

    return tensors_resized["image"] / 255.0, tensors_resized["mask"], tensors_resized["proposals"], \
           tensors_resized["proposals"], mask_void

  def get_support_indices(self, index, sequence):
    # First frame would be the reference frame during testing
    # sampling_window = self.temporal_window if self.is_train else self.temporal_window - 1
    sampling_window = self.temporal_window
    # index should be start index of the clip
    if self.is_train:
      index_range = np.arange(index, min(self.num_frames[sequence],
                                         (index + max(self.max_temporal_gap, sampling_window))))
    else:
      index_range = [0]
      index_range = np.append(index_range,
                              np.arange(index, min(self.num_frames[sequence], (index + sampling_window)))
                              )
      # index_range = np.arange(index, min(self.num_frames[sequence], (index + sampling_window)))

    support_indices = np.random.choice(index_range, min(self.temporal_window, len(index_range)), replace=False)
    support_indices = np.sort(np.append(support_indices, np.repeat([index],
                                                                   self.temporal_window - len(support_indices))))

    # print(support_indices)
    return support_indices


class DAVISSimilarity(DAVIS16):
  def __init__(self, root, imset='2017/train.txt', resolution='480p', is_train=False,
               random_instance=False, num_classes=2, crop_size=None,temporal_window=8, min_temporal_gap=2,
               max_temporal_gap=8, resize_mode=ResizeMode.FIXED_SIZE, proposal_dir=None, augmentors=None):
    # FIXME: this flag is a hack to check if it is a multi class training
    self.multi = (num_classes != 2)
    super(DAVISSimilarity, self).__init__(root, imset, is_train=is_train, crop_size=crop_size,
                                  temporal_window=temporal_window, random_instance=random_instance,
                                  resize_mode=resize_mode, proposal_dir=proposal_dir,
                                  max_temporal_gap=max_temporal_gap, min_temporal_gap=1, num_classes=10)

  def __getitem__(self, item):
    input_dict = super(DAVISSimilarity, self).__getitem__(item)
    one_hot_masks = [get_one_hot_vectors(input_dict['target'][0, i], self.num_classes)[:, np.newaxis, :, :]
                     for i in range(len(input_dict['target'][0]))]
    # one_hot_masks = get_one_hot_vectors(input_dict['target'][0, 0], self.num_classes)

    # add one hot vector masks as extra target
    # input_dict['target_extra'] = {'similarity': np.concatenate(one_hot_masks, axis=1).astype(np.uint8),
    #                               'similarity_raw_mask': input_dict['target']}
    input_dict['target_extra'] = {'similarity_ref': np.concatenate(one_hot_masks, axis=1).astype(np.uint8),
                                  'similarity_raw_mask': input_dict['target'][:, 1:]}
    if not self.multi:
      input_dict['target'] = (input_dict['target'] != 0).astype(np.uint8)

    return input_dict
