import glob
import os
import random

import numpy as np

from datasets.DAVIS import DAVIS
from utils.Resize import ResizeMode


class DAVISProposalGuidance(DAVIS):
  def __init__(self, root, imset='2017/train.txt', resolution='480p', is_train=False,
               random_instance=False, num_classes=2, crop_size=None,temporal_window=5, min_temporal_gap=2,
               max_temporal_gap=8, resize_mode=ResizeMode.FIXED_SIZE, proposal_dir=None, augmentors=None):
    super(DAVISProposalGuidance, self).__init__(root, imset, resolution, is_train,
               random_instance, num_classes, crop_size,temporal_window, min_temporal_gap,
               max_temporal_gap, resize_mode, proposal_dir, augmentors)

  def __getitem__(self, item):
    input_dict = super(DAVISProposalGuidance, self).__getitem__(item)
    input_dict['masks_guidance'][:, 1:-1] = input_dict['proposals'][:,1:-1]
    return input_dict


class DAVISProposalGuidanceEval(DAVISProposalGuidance):
  def __init__(self, root, imset='2017/val.txt', is_train=False, crop_size=None, temporal_window=5,
               random_instance=False, resize_mode=ResizeMode.FIXED_SIZE, proposal_dir=None):
    super(DAVISProposalGuidance, self).__init__(root, imset, is_train=is_train, crop_size=crop_size,
                                    temporal_window=temporal_window, random_instance=random_instance,
                                    resize_mode=resize_mode, proposal_dir=proposal_dir)

  def set_video_id(self, video):
    self.current_video = video
    self.start_index = self.get_start_index(video)
    self.img_list = list(glob.glob(os.path.join(self.image_dir, self.current_video, '*.jpg')))
    self.img_list.sort()

  def get_video_ids(self):
    # shuffle the list for training
    return random.sample(self.videos, len(self.videos)) if self.is_train else self.videos

  def __len__(self):
    if self.current_video is None:
      raise IndexError("set a video before enumerating through the dataset")
    else:
      return self.num_frames[self.current_video]

  def get_support_indices(self, index, sequence):
    if index == 0:
      support_indices = np.repeat([index], self.temporal_window)
    elif (index - self.temporal_window) < 0:
      support_indices = np.repeat([index], abs(index - self.temporal_window))
      support_indices = np.append(support_indices, np.array([index-1]))
      indices_to_sample = np.array(range((index - self.temporal_window), index-1))
      indices_to_sample = indices_to_sample[indices_to_sample>=0]
      support_indices = np.append(support_indices, indices_to_sample)
      support_indices.sort()
    else:
      support_indices = np.array(range((index - self.temporal_window), index))

    return support_indices.astype(np.int)

  def get_current_sequence(self, img_file):
    return self.current_video


class DAVISProposalGuidanceInfer(DAVISProposalGuidanceEval):
  def __init__(self, root, imset='2017/val.txt', is_train=False, crop_size=None, temporal_window=5,
               random_instance=False, resize_mode=ResizeMode.FIXED_SIZE, proposal_dir=None):
    super(DAVISProposalGuidanceInfer, self).__init__(root, imset, is_train=is_train, crop_size=crop_size,
                                    temporal_window=temporal_window, random_instance=random_instance,
                                    resize_mode=resize_mode, proposal_dir=proposal_dir)
    self.max_temporal_gap = 12

  def get_support_indices(self, index, sequence):
    if index == 0:
      support_indices = np.repeat([index], self.temporal_window)
    elif (index - self.temporal_window) < 0:
      support_indices = np.repeat([0], abs(index - self.temporal_window + 1))
      support_indices = np.append(support_indices, np.array([index-1, index]))
      indices_to_sample = np.array(range((index - self.temporal_window), index-1))
      indices_to_sample = indices_to_sample[indices_to_sample>=0]
      support_indices = np.append(support_indices, indices_to_sample)
      support_indices.sort()
    else:
      support_indices = np.array([0, index-1, index])
      for i in range(index // self.max_temporal_gap, 0):
        if i*self.max_temporal_gap not in support_indices and len(support_indices) < self.temporal_window:
          support_indices = np.append(support_indices, [i*self.max_temporal_gap])

      sample_indices = np.setdiff1d(np.array(list(range(0, max(0, index-2)))), support_indices)
      if len(support_indices)< self.temporal_window:
        support_indices = np.append(support_indices,
                                                 np.random.choice(sample_indices,
                                                                  self.temporal_window - len(support_indices)))
      # support_indices = np.array([0])
      # support_indices = np.append(support_indices, np.array(list(range(index - self.temporal_window + 2, index + 1))))
      support_indices.sort()

    print("support indices are {}".format(support_indices))
    return np.abs(support_indices)

  def __getitem__(self, index):
    input_dict = super(DAVISProposalGuidanceInfer, self).__getitem__(index)
    support_indices = self.get_support_indices(index, input_dict['info']['name'])
    input_dict['info']['support_indices'] = support_indices
    return input_dict