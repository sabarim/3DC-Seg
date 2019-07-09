import glob
import os
import random

import numpy as np
from PIL import Image
from scipy.misc import imresize
from torch.utils import data


class DAVIS(data.Dataset):
  def __init__(self, root, imset='2017/train.txt', resolution='480p', is_train=False,
               random_instance=False, num_classes=2, crop_size=None,temporal_window=3, min_temporal_gap=2,
               max_temporal_gap=8):
    self.current_video = None
    self.root = root
    self.num_classes = num_classes
    self.crop_size = crop_size
    self.is_train = is_train
    self.random_instance = random_instance and is_train
    # remove proposals that match the ground truth randomly. This can emulate missing proposals/ matches
    self.mask_dir = os.path.join(root, 'Annotations', resolution)
    self.image_dir = os.path.join(root, 'JPEGImages', resolution)
    _imset_dir = os.path.join(root, 'ImageSets')
    _imset_f = os.path.join(_imset_dir, imset)
    self.max_proposals = 20
    self.start_index = None
    self.temporal_window = temporal_window
    self.min_temporal_gap = min_temporal_gap
    self.max_temporal_gap = max_temporal_gap

    self.videos = []
    self.num_frames = {}
    self.num_objects = {}
    self.shape = {}
    self.img_list = []
    self.create_img_list(_imset_f)

  def create_img_list(self, _imset_f):
    with open(os.path.join(_imset_f), "r") as lines:
      for line in lines:
        _video = line.rstrip('\n')
        self.videos.append(_video)
        self.num_frames[_video] = len(glob.glob(os.path.join(self.image_dir, _video, '*.jpg')))
        _mask = np.array(Image.open(os.path.join(self.mask_dir, _video, '00000.png')).convert("P"))
        self.num_objects[_video] = np.max(_mask)
        self.shape[_video] = np.shape(_mask)
        self.img_list += list(glob.glob(os.path.join(self.image_dir, _video, '*.jpg')))

  def set_video_id(self, video):
    self.current_video = video
    self.start_index = self.get_start_index(video)

  def get_video_ids(self):
    # shuffle the list for training
    return random.sample(self.videos, len(self.videos)) if self.is_train else self.videos

  def get_start_index(self, video):
    num_frames = self.num_frames[self.current_video]
    start_frame = 0

    # choose a start frame which has atleast one object visible
    while self.is_train:
      assert self.bptt_len < num_frames
      start_frame = random.randint(0, num_frames - self.bptt_len - 1)
      mask_file = os.path.join(self.mask_dir, video, '{:05d}.png'.format(start_frame))  # allways return first frame mask
      raw_mask = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
      if raw_mask.sum() > 0:
        break

    return start_frame

  def __len__(self):
    return len(self.img_list)

  def read_frame(self, shape, video, f, instance_id=None):
    # use a blend of both full random instance as well as the full object
    img_file = os.path.join(self.image_dir, video, '{:05d}.jpg'.format(f))
    raw_frames = imresize(np.array(Image.open(img_file).convert('RGB')) / 255., shape) / 255.0
    try:
      mask_file = os.path.join(self.mask_dir, video, '{:05d}.png'.format(f))  # allways return first frame mask
      raw_mask = imresize(np.array(Image.open(mask_file).convert('P'), dtype=np.uint8), shape,
                          interp="nearest")
    except:
      mask_file = os.path.join(self.mask_dir, video, '00000.png')
      raw_mask = imresize(np.array(Image.open(mask_file).convert('P'), dtype=np.uint8), shape,
                          interp="nearest")

    raw_masks = (raw_mask == instance_id).astype(np.uint8) if instance_id is not None else raw_mask

    return raw_frames, raw_masks

  def __getitem__(self, index):
    img_file = self.img_list[index]
    sequence = self.get_current_sequence(img_file)
    info = {}
    info['name'] = sequence
    info['num_frames'] = self.num_frames[sequence]
    num_objects = self.num_objects[sequence]
    info['num_objects'] = num_objects
    info['shape'] = self.shape[sequence]
    obj_id = 1
    index = int(os.path.splitext(os.path.basename(img_file))[0])

    # retain original shape
    # shape = self.shape[self.current_video] if not (self.is_train and self.MO) else self.crop_size
    shape = self.shape[sequence] if self.crop_size is None else self.crop_size
    support_indices = self.get_support_indices(index, sequence)
    info['support_indices'] = support_indices
    th_frames = []
    th_masks = []
    instance_id = np.random.choice(np.array(range(1,num_objects+1))) if self.random_instance else None
    # add the current index and the previous frame with respect to the max index in supporting frame
    for i in np.sort(support_indices):
      raw_frame, raw_mask = self.read_frame(shape, sequence, i, instance_id)

      # padding size to be divide by 32
      h, w = raw_mask.shape
      new_h = h + 32 - h % 32
      new_w = w + 32 - w % 32
      # print(new_h, new_w)
      lh, uh = (new_h - h) / 2, (new_h - h) / 2 + (new_h - h) % 2
      lw, uw = (new_w - w) / 2, (new_w - w) / 2 + (new_w - w) % 2
      lh, uh, lw, uw = int(lh), int(uh), int(lw), int(uw)
      pad_masks = np.pad(raw_mask, ((lh, uh), (lw, uw)), mode='constant')
      pad_frames = np.pad(raw_frame, ((lh, uh), (lw, uw), (0, 0)), mode='constant')
      info['pad'] = ((lh, uh), (lw, uw))

      th_frames.append(np.transpose(pad_frames, (2, 0, 1))[:, np.newaxis])
      th_masks.append(pad_masks[np.newaxis, np.newaxis])

    target = th_masks[-1][0]
    th_masks[-1] = np.zeros_like(th_masks[-1])
    return {'images': np.concatenate(th_frames, axis=1), 'masks_guidance':np.concatenate(th_masks, axis=1), 'info': info,
            'target': target}

  def get_current_sequence(self, img_file):
    sequence = img_file.split("/")[-2]
    return sequence

  def get_support_indices(self, index, sequence):
    support_indices = [i for i in range(self.num_frames[sequence])
                       if abs(index - i) > self.min_temporal_gap and abs(index - i) < self.max_temporal_gap]
    support_indices = np.random.choice(support_indices, self.temporal_window, replace=False)
    support_indices = np.append(support_indices, np.array([index, np.max(support_indices) - 1]))
    return support_indices


class DAVISEval(DAVIS):
  def __init__(self, root, imset='2017/val.txt', is_train=False, crop_size=None, temporal_window=5,
               random_instance=False):
    super(DAVISEval, self).__init__(root, imset, is_train=is_train, crop_size=crop_size,
                                    temporal_window=temporal_window, random_instance=random_instance)

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




