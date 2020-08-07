import random
from abc import abstractmethod

import cv2
import numpy as np
from PIL import Image
from imageio import imread
from utils.Resize import resize, ResizeMode
from torch.utils.data import Dataset


def list_to_dict(list):
  """

  :param list: input list
  :return: converted dictionary
  """
  result_dict = {str(i): val for i, val in enumerate(list)}
  return result_dict


class BaseDataset(Dataset):
  def __init__(self, root, mode='train', resize_mode=None, resize_shape=None):
    self.resize_mode = ResizeMode(resize_mode)
    self.resize_shape = resize_shape
    self.mode = mode
    self.root = root
    self.samples = []
    self.create_sample_list()

  # Override in case tensors have to be normalised
  def normalise(self, tensors):
    tensors['images'] = tensors['images'].astype(np.float32) / 255.0
    return tensors

  def is_train(self):
    return self.mode == "train"

  def pad_tensors(self, tensors_resized):
    h, w = tensors_resized["images"].shape[:2]
    new_h = h + 32 - h % 32 if h % 32 > 0 else h
    new_w = w + 32 - w % 32 if w % 32 > 0 else w
    lh, uh = (new_h - h) / 2, (new_h - h) / 2 + (new_h - h) % 2
    lw, uw = (new_w - w) / 2, (new_w - w) / 2 + (new_w - w) % 2
    lh, uh, lw, uw = int(lh), int(uh), int(lw), int(uw)

    padded_tensors = {}
    for key, tensor in tensors_resized.items():
      if tensor.ndim == 2:
        tensor = tensor[..., None]
      assert tensor.ndim == 3
      padded_tensors[key] = np.pad(tensor,
                                   ((lh, uh), (lw, uw), (0, 0)),
                                   mode='constant')

    return padded_tensors

  def read_sample(self, sample):
    images = map(imread, sample['images'])
    # targets = map(imread, sample['targets'])
    targets = map(lambda x: np.array( Image.open(x).convert('P'), dtype=np.uint8), sample['targets'])

    images = list_to_dict(images)
    images = resize(images, self.resize_mode, self.resize_shape)
    images = np.stack(images.values())

    targets = list_to_dict(targets)
    targets = resize(targets, self.resize_mode, self.resize_shape)
    targets = np.stack(targets.values())

    data = {"images": images, "targets": targets}
    for key, val in sample.items():
      if key in ['images', 'targets']:
        continue
      if key in data:
        data[key] += [val]
      else:
        data[key] = [val]
    return data

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    sample = self.samples[idx]
    tensors_resized = self.read_sample(sample)

    padded_tensors = self.pad_tensors(tensors_resized)

    padded_tensors = self.normalise(padded_tensors)

    return {"images": [np.transpose(padded_tensors['img1'], (2, 0, 1)).astype(np.float32),
                       np.transpose(padded_tensors['img2'], (2, 0, 1)).astype(np.float32)],
            "target": {"flow": np.transpose(padded_tensors['flow'], (2, 0, 1)).astype(np.float32)}, 'info': {}}

  @abstractmethod
  def create_sample_list(self):
    pass


class VideoDataset(BaseDataset):
  def __init__(self, root, mode='train', resize_mode=None, resize_shape=None, tw=8, max_temporal_gap=8, num_classes=2):
    self.tw = tw
    self.max_temporal_gap = max_temporal_gap
    self.num_classes = num_classes

    self.videos = []
    self.num_frames = {}
    self.num_objects = {}
    self.shape = {}

    self.current_video = None
    self.start_index = None
    super(VideoDataset, self).__init__(root, mode, resize_mode, resize_shape)

  def set_video_id(self, video):
    self.current_video = video
    self.start_index = self.get_start_index(video)

  def get_video_ids(self):
    # shuffle the list for training
    return random.sample(self.videos, len(self.videos)) if self.is_train() else self.videos

  def get_start_index(self, video):
    start_frame = 0
    return start_frame

  def pad_tensors(self, tensors_resized):
    h, w = tensors_resized["images"].shape[1:3]
    new_h = h + 32 - h % 32 if h % 32 > 0 else h
    new_w = w + 32 - w % 32 if w % 32 > 0 else w
    lh, uh = (new_h - h) / 2, (new_h - h) / 2 + (new_h - h) % 2
    lw, uw = (new_w - w) / 2, (new_w - w) / 2 + (new_w - w) % 2
    lh, uh, lw, uw = int(lh), int(uh), int(lw), int(uw)

    padded_tensors = tensors_resized.copy()
    keys = ['images', 'targets']

    for key in keys:
      pt = []
      t = tensors_resized[key]
      if t.ndim == 3:
        t = t[..., None]
      assert t.ndim == 4
      padded_tensors[key] = np.pad(t,
                   ((0,0),(lh, uh), (lw, uw), (0, 0)),
                   mode='constant')

    return padded_tensors

  def __getitem__(self, idx):
    sample = self.samples[idx]
    tensors_resized = self.read_sample(sample)

    padded_tensors = self.pad_tensors(tensors_resized)

    padded_tensors = self.normalise(padded_tensors)

    return {"images": np.transpose(padded_tensors['images'], (3, 0, 1, 2)).astype(np.float32),
            "target": {"mask": np.transpose(padded_tensors['targets'], (3, 0, 1, 2)).astype(np.float32)},
            'info': padded_tensors['info']}

  @abstractmethod
  def get_support_indices(self, index, sequence):
    pass
