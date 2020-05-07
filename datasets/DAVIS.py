import glob
import os
import pickle
import random

import numpy as np
from PIL import Image
from torch.utils import data

from util import create_object_id_mapping
from util import top_n_predictions_maskrcnn, filter_by_category
from utils.Resize import ResizeMode, resize

PASCAL_VOC_PATH = '/globalwork/data/pascal_voc/'


class DAVIS(data.Dataset):
  def __init__(self, root, imset='2017/train.txt', resolution='480p', is_train=False,
               random_instance=False, num_classes=2, crop_size=None,temporal_window=5, min_temporal_gap=2,
               max_temporal_gap=8, resize_mode=ResizeMode.FIXED_SIZE, proposal_dir=None, augmentors=None):
    self.current_video = None
    self.root = root
    self.num_classes = num_classes
    self.crop_size = crop_size
    self.is_train = is_train
    self.random_instance = random_instance
    _imset_f = self.set_paths(imset, resolution, root)
    self.max_proposals = 20
    self.start_index = None
    self.temporal_window = temporal_window
    self.min_temporal_gap = min_temporal_gap
    self.max_temporal_gap = max_temporal_gap
    self.resize_mode = ResizeMode(resize_mode)
    self.proposal_dir = proposal_dir

    self.videos = []
    self.num_frames = {}
    self.num_objects = {}
    self.shape = {}
    self.shape480p = {}
    self.img_list = []
    self.reference_list = []
    self.create_img_list(_imset_f)
    # max width of a video which can be used for padding during training
    self.max_w = max([w for (h, w) in self.shape.values()])
    self.min_w = min([w for (h, w) in self.shape.values()])
    self.max_h = max([h for (h, w) in self.shape.values()])
    self.min_h = min([h for (h, w) in self.shape.values()])
    print("Max image width {} : Max image height {}\n"
          "Min image width: {} MIN Image height: {}".format(self.max_w, self.max_h, self.min_w, self.min_h))
    # self.occluders = load_occluders(PASCAL_VOC_PATH)

  def set_paths(self, imset, resolution, root):
    print(resolution)
    self.mask_dir = os.path.join(root, 'Annotations_unsupervised', resolution)
    self.image_dir = os.path.join(root, 'JPEGImages', resolution)
    _imset_dir = os.path.join(root, 'ImageSets')
    _imset_f = os.path.join(_imset_dir, imset)
    return _imset_f

  def create_img_list(self, _imset_f):
    with open(os.path.join(_imset_f), "r") as lines:
      for line in lines:
        _video = line.rstrip('\n')
        self.videos.append(_video)
        self.num_frames[_video] = len(glob.glob(os.path.join(self.image_dir, _video, '*.jpg')))
        _mask_file = os.path.join(self.mask_dir, _video, '00000.png')
        if os.path.exists(_mask_file):
          _mask = np.array(Image.open(os.path.join(self.mask_dir, _video, '00000.png')).convert("P"))
          self.num_objects[_video] = np.max(_mask)
          self.shape[_video] = np.shape(_mask)
          self.shape480p[_video] = np.shape(
            np.array(Image.open(os.path.join(self.mask_dir.replace('Full-Resolution', '480p'),
                                             _video, '00000.png')).convert("P")))
        else:
          _image = np.array(Image.open(os.path.join(self.image_dir, _video, '00000.jpg')).convert("P"))
          self.num_objects[_video] = -1
          self.shape[_video] = np.shape(_image.shape[:2])
          self.shape480p[_video] = np.array(Image.open(os.path.join(self.image_dir.replace('Full-Resolution', '480p'), _video,
                                                                    '00000.png')).convert("RGB")).shape[:2]

        self.img_list += list(glob.glob(os.path.join(self.image_dir, _video, '*.jpg')))

      if self.is_train:
        self.reference_list = [f for f in self.img_list if self.img_list if np.sum(
          np.array(Image.open(f.replace('JPEGImages', 'Annotations_unsupervised').replace('.jpg', '.png'))))]

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
    return len(self.reference_list) if self.is_train else len(self.img_list)

  def read_frame(self, shape, video, f, instance_id=None, support_indices=None):
    # use a blend of both full random instance as well as the full object
    img_file = os.path.join(self.image_dir, video, '{:05d}.jpg'.format(f))
    raw_frames = np.array(Image.open(img_file).convert('RGB')) / 255.
    try:
      mask_file = os.path.join(self.mask_dir, video, '{:05d}.png'.format(f))  # allways return first frame mask
      # raw_mask = imresize(np.array(Image.open(mask_file).convert('P'), dtype=np.uint8), shape,
      #                     interp="nearest")
      raw_mask = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
    except:
      mask_file = os.path.join(self.mask_dir, video, '00000.png')
      raw_mask = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)

    mask_void = (raw_mask == 255).astype(np.uint8)
    raw_mask[raw_mask == 255] = 0
    raw_masks = (raw_mask == instance_id).astype(np.uint8) if instance_id is not None else raw_mask
    num_proposals, raw_proposals, proposal_mask = self.read_proposals(video, f, raw_masks)
    # if self.is_train:
    #   [raw_frames, raw_masks] = do_occ_aug(self.occluders, [raw_frames, raw_masks])
    tensors_resized = resize({"image":raw_frames, "mask":raw_masks, "proposals": proposal_mask.astype(np.uint8)},
                             self.resize_mode, shape)

    return tensors_resized["image"] / 255.0, tensors_resized["mask"], tensors_resized["proposals"], raw_proposals, mask_void

  def read_proposals(self, video, f, gt_mask):
    proposal_file = os.path.join(self.proposal_dir, video, '{:05d}.pickle'.format(f))
    proposals = pickle.load(open(proposal_file, 'rb'))
    proposals = top_n_predictions_maskrcnn(proposals, self.max_proposals)
    proposals, object_mapping = self.filter_proposals(gt_mask, proposals)

    raw_proposals = {'scores':np.zeros(self.max_proposals), 'labels':np.zeros(self.max_proposals)}
    if len(proposals['mask']) == 0:
      print("WARN: no proposals found in {}".format(proposal_file))
    num_proposals = len(proposals['mask'])
    raw_proposals['scores'][:len(proposals['mask'])] = proposals['scores']
    raw_proposals['labels'][:len(proposals['mask'])] = proposals['labels']
    proposal_mask = self.get_proposal_maks(proposals['scores'], proposals['mask'])

    return num_proposals, raw_proposals, proposal_mask

  def filter_proposals(self, gt_mask, proposals):
    # proposal_mask = self.get_proposal_maks(proposals['scores'], proposals['mask'])
    object_mapping = create_object_id_mapping(gt_mask, proposals['mask'][:, 0].data.cpu().numpy())
    # find the object category of the best overlapped object
    best_overlap_ids = np.setdiff1d(list(object_mapping.values()), [-1])
    if len(best_overlap_ids) > 0:
      overlap_categories = proposals['labels'][best_overlap_ids]
      proposals_filtered = filter_by_category(proposals, overlap_categories)
    else:
      proposals_filtered = proposals
    return proposals_filtered, object_mapping

  def get_proposal_maks(self, proposal_scores, raw_proposals):
    proposals_weighted = (raw_proposals[:, 0].float() * proposal_scores[:, np.newaxis, np.newaxis]).data.cpu().numpy()
    # add background mask before conputing the argmax
    proposals_weighted = np.concatenate(((np.sum(proposals_weighted, axis=0, keepdims=True) == 0).astype(np.int),
                                         proposals_weighted),
                                        axis=0)
    proposal_mask = np.argmax(proposals_weighted, axis=0)
    return proposal_mask

  def __getitem__(self, index):
    img_file = self.reference_list[index] if self.is_train else self.img_list[index]
    sequence = self.get_current_sequence(img_file)
    info = {}
    info['name'] = sequence
    info['num_frames'] = self.num_frames[sequence]
    num_objects = self.num_objects[sequence]
    info['num_objects'] = num_objects
    info['shape'] = self.shape[sequence]
    info['shape480p'] = self.shape480p[sequence] if sequence in self.shape480p else -1
    obj_id = 1
    index = int(os.path.splitext(os.path.basename(img_file))[0])

    # retain original shape
    # shape = self.shape[self.current_video] if not (self.is_train and self.MO) else self.crop_size
    shape = self.shape[sequence] if self.crop_size is None else self.crop_size
    support_indices = self.get_support_indices(index, sequence)
    info['support_indices'] = support_indices
    th_frames = []
    th_masks = []
    th_mask_void = []
    th_proposals = []
    th_raw_proposals = []
    th_pc = []
    th_ps = []
    instance_id = np.random.choice(np.array(range(1,num_objects+1))) if self.random_instance else None
    # add the current index and the previous frame with respect to the max index in supporting frame
    for i in np.sort(support_indices):
      raw_frame, raw_mask, proposals, raw_proposals, mask_void = \
        self.read_frame(shape, sequence, i, instance_id, support_indices)

      # padding size to be divide by 32
      h, w = raw_mask.shape
      new_h = h + 32 - h % 32 if h % 32 > 0 else h
      new_w = w + 32 - w % 32 if w % 32 > 0 else w
      # print(new_h, new_w)
      lh, uh = (new_h - h) / 2, (new_h - h) / 2 + (new_h - h) % 2
      lw, uw = (new_w - w) / 2, (new_w - w) / 2 + (new_w - w) % 2
      lh, uh, lw, uw = int(lh), int(uh), int(lw), int(uw)
      pad_masks = np.pad(raw_mask, ((lh, uh), (lw, uw)), mode='constant')
      pad_mask_void = np.pad(mask_void, ((lh, uh), (lw, uw)), mode='constant')
      pad_proposals = np.pad(proposals, ((lh, uh), (lw, uw)), mode='constant')
      pad_raw_proposals = np.pad(raw_proposals, ((lh, uh), (lw, uw)), mode='constant')
      pad_frames = np.pad(raw_frame, ((lh, uh), (lw, uw), (0, 0)), mode='constant')
      info['pad'] = ((lh, uh), (lw, uw))

      th_frames.append(np.transpose(pad_frames, (2, 0, 1))[:, np.newaxis])
      th_masks.append(pad_masks[np.newaxis, np.newaxis])
      th_mask_void.append(pad_mask_void[np.newaxis, np.newaxis])
      th_proposals.append(pad_proposals[np.newaxis, np.newaxis])
      th_raw_proposals.append(pad_raw_proposals[np.newaxis, np.newaxis])
      # th_pc.append(proposal_categories[np.newaxis])
      # th_ps.append(proposal_scores[np.newaxis])

    target = th_masks[-1][0]
    th_masks_raw = th_masks.copy()
    th_masks[-1] = np.zeros_like(th_masks[-1])
    masks_guidance = np.concatenate(th_masks, axis=1)
    # remove masks with some probability to make sure that the network can focus on intermediate frames
    # if self.is_train and np.random.choice([True, False], 1, p=[0.15,0.85]):
    #   masks_guidance[0, -2] = np.zeros(masks_guidance.shape[2:])

    #Can be used for setting proposals if desired, but for now it isn't neccessary and will be ignored
    proposals = np.concatenate(th_proposals, axis=1)
    return {'images': np.concatenate(th_frames, axis=1), 'masks_guidance':masks_guidance, 'info': info,
            'target': target, "proposals": proposals,
            "raw_proposals": np.concatenate(th_raw_proposals, axis=1), 'raw_masks': np.concatenate(th_masks_raw, axis=1)}

  def get_current_sequence(self, img_file):
    sequence = img_file.split("/")[-2]
    return sequence

  def get_support_indices(self, index, sequence):
    if index == 0:
      support_indices = np.repeat([0], self.temporal_window)
    else:
      if self.is_train:
        support_indices = np.array([index])
      else:
        support_indices = np.array([index, index - 1])

      sample_indices = np.array(list(range(max(0, abs(index - self.max_temporal_gap)),
                                            max(0, index-self.min_temporal_gap))))
      num_indices = self.temporal_window - len(support_indices)
      if len(sample_indices) >= num_indices:
        support_indices = np.append(support_indices, np.random.choice(sample_indices, num_indices, replace=False))
      else:
        support_indices = np.append(support_indices, np.random.choice(sample_indices, num_indices, replace=True))
        # support_indices = np.append(np.repeat([max(0, index-1)], self.temporal_window - len(support_indices)),
        #                             support_indices)
    support_indices.sort()
    # print(index, support_indices)
    return support_indices.astype(np.int)


class DAVISEval(DAVIS):
  def __init__(self, root, imset='2017/val.txt', is_train=False, crop_size=None, temporal_window=5,
               random_instance=False, resize_mode=ResizeMode.FIXED_SIZE, proposal_dir=None):
    super(DAVISEval, self).__init__(root, imset, is_train=is_train, crop_size=crop_size,
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


class DAVISInfer(DAVISEval):
  def __init__(self, root, imset='2017/val.txt', is_train=False, crop_size=None, temporal_window=5,
               random_instance=False, resize_mode=ResizeMode.FIXED_SIZE, proposal_dir=None):
    super(DAVISInfer, self).__init__(root, imset, is_train=is_train, crop_size=crop_size,
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
    input_dict = super(DAVISInfer, self).__getitem__(index)
    support_indices = self.get_support_indices(index, input_dict['info']['name'])
    input_dict['info']['support_indices'] = support_indices
    return input_dict




