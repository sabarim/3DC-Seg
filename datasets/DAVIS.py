import glob
import os
import random
import pickle
import numpy as np
from PIL import Image
from scipy.misc import imresize
from torch.utils import data

from util import top_n_predictions_maskrcnn
from utils.Resize import ResizeMode, resize


class DAVIS(data.Dataset):
  def __init__(self, root, imset='2017/train.txt', resolution='480p', is_train=False,
               random_instance=False, num_classes=2, crop_size=None,temporal_window=5, min_temporal_gap=2,
               max_temporal_gap=8, resize_mode=ResizeMode.FIXED_SIZE, proposal_dir=None):
    self.current_video = None
    self.root = root
    self.num_classes = num_classes
    self.crop_size = crop_size
    self.is_train = is_train
    self.random_instance = random_instance and is_train
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
    self.img_list = []
    self.create_img_list(_imset_f)
    # max width of a video which can be used for padding during training
    self.max_w = max([w for (h, w) in self.shape.values()])

  def set_paths(self, imset, resolution, root):
    self.mask_dir = os.path.join(root, 'Annotations', resolution)
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
    raw_frames = np.array(Image.open(img_file).convert('RGB')) / 255.
    try:
      mask_file = os.path.join(self.mask_dir, video, '{:05d}.png'.format(f))  # allways return first frame mask
      raw_mask = imresize(np.array(Image.open(mask_file).convert('P'), dtype=np.uint8), shape,
                          interp="nearest")
    except:
      mask_file = os.path.join(self.mask_dir, video, '00000.png')
      raw_mask = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)

    raw_masks = (raw_mask == instance_id).astype(np.uint8) if instance_id is not None else raw_mask
    tensors_resized = resize({"image":raw_frames, "mask":raw_masks}, self.resize_mode, shape)

    return tensors_resized["image"] / 255.0, tensors_resized["mask"]

  def read_proposals(self, video, f, gt_mask):
    proposal_file = os.path.join(self.proposal_dir, video, '{:05d}.pickle'.format(f))
    proposals = pickle.load(open(proposal_file, 'rb'))
    proposals = top_n_predictions_maskrcnn(proposals, self.max_proposals)
    raw_proposals = np.zeros((self.max_proposals,) + tuple(gt_mask.shape))
    proposal_categories = np.zeros(self.max_proposals)
    proposal_scores = np.zeros(self.max_proposals)
    if len(proposals['mask']) > 0:
      raw_proposals[:len(proposals['mask'])] = proposals['mask'][:, 0].data.cpu().numpy()
    else:
      print("WARN: no proposals found in {}".format(proposal_file))
    num_proposals = len(proposals['mask'])
    proposal_scores[:len(proposals['mask'])] = proposals['scores']
    proposal_categories[:len(proposals['mask'])] = proposals['labels']

    # remove gt proposal if required
    # if self.remove_gt_proposal:# and np.random.choice([True, False], p=[0.3, 0.7]):
    #   proposal_selected = get_best_match(torch.from_numpy(gt_mask).unsqueeze(0), torch.from_numpy(raw_proposals),
    #                                      num_proposals=torch.tensor(num_proposals, dtype=torch.int),
    #                                      object_categories=torch.from_numpy(proposal_categories))
    #   if proposal_selected is not None:
    #     raw_proposals[proposal_selected[2]] = np.zeros_like(raw_proposals[proposal_selected[2]])
    return num_proposals

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
    if index == 0:
      support_indices = np.repeat([0], self.temporal_window)
    else:
      support_indices = np.array(list(range(max(0, abs(index - self.max_temporal_gap)),
                                            max(0, index-self.min_temporal_gap))))
      if len(support_indices) >= self.temporal_window - 2:
        support_indices = np.random.choice(support_indices, self.temporal_window - 2, replace=False)
      else:
        support_indices = np.append(np.repeat([max(0, index-1)], self.temporal_window - 2 - len(support_indices)),
                                    support_indices)
      support_indices = np.append(support_indices, np.array([index, index-1]))
    support_indices.sort()
    # print(index, support_indices)
    return support_indices.astype(np.int)


class DAVISEval(DAVIS):
  def __init__(self, root, imset='2017/val.txt', is_train=False, crop_size=None, temporal_window=5,
               random_instance=False, resize_mode=ResizeMode.FIXED_SIZE):
    super(DAVISEval, self).__init__(root, imset, is_train=is_train, crop_size=crop_size,
                                    temporal_window=temporal_window, random_instance=random_instance,
                                    resize_mode=resize_mode)

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
               random_instance=False, resize_mode=ResizeMode.FIXED_SIZE):
    super(DAVISInfer, self).__init__(root, imset, is_train=is_train, crop_size=crop_size,
                                    temporal_window=temporal_window, random_instance=random_instance,
                                    resize_mode=resize_mode)

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
      support_indices = np.array([0])
      support_indices = np.append(support_indices, np.array(list(range(index - self.temporal_window + 2, index + 1))))
      # max frames that could be sampled based on the temporal window
      # max_frames_tw = int(self.temporal_window - (index // self.temporal_window) - 2)
      # if max_frames_tw > 0:
        # sample previous frames if enough support frames with the required temporal gap are not available.
        # support_indices = np.append(support_indices, list(range((index - max_frames_tw - 1), index - 1)))

      # for i in range(self.temporal_window - len(support_indices)):
        #support_indices = np.append(support_indices, np.array([index - (self.temporal_window * (i+1))]))
        # support_indices = np.append(support_indices, np.array([(self.temporal_window * i)]))

      support_indices.sort()

    # print("support indices are {}".format(support_indices))
    return np.abs(support_indices)

  def __getitem__(self, index):
    input_dict = super(DAVISInfer, self).__getitem__(index)
    support_indices = self.get_support_indices(index, input_dict['info']['name'])
    input_dict['info']['support_indices'] = support_indices
    return input_dict




