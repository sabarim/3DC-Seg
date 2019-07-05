import glob
import os
import numpy as np
import torch
import random
import pickle
from PIL import Image
from scipy.misc import imresize
from scipy.ndimage import distance_transform_edt
from skimage import io
from torch.utils import data

from utils import top_n_predictions_maskrcnn, get_best_match


class DAVIS(data.Dataset):
  def __init__(self, root, imset='2016/val.txt', resolution='480p', multi_object=False, is_train=False,
               random_instance=False, num_classes=2, crop_size=(513, 513),
               proposal_dir="/globalwork/mahadevan/mywork/data/training/pytorch/forwarded/maskrcnn/thresh-0/",
               bptt_len = 12, remove_gt_proposal = False):
    self.current_video = None
    self.root = root
    self.num_classes = num_classes
    self.crop_size = crop_size
    self.is_train = is_train
    self.random_instance = random_instance and is_train
    # remove proposals that match the ground truth randomly. This can emulate missing proposals/ matches
    self.remove_gt_proposal = remove_gt_proposal and is_train
    self.mask_dir = os.path.join(root, 'Annotations', resolution)
    self.image_dir = os.path.join(root, 'JPEGImages', resolution)
    _imset_dir = os.path.join(root, 'ImageSets')
    _imset_f = os.path.join(_imset_dir, imset)
    self.proposal_dir = proposal_dir
    self.max_proposals = 20
    self.bptt_len = bptt_len
    self.start_index = None

    self.videos = []
    self.num_frames = {}
    self.num_objects = {}
    self.shape = {}
    with open(os.path.join(_imset_f), "r") as lines:
      for line in lines:
        _video = line.rstrip('\n')
        self.videos.append(_video)
        self.num_frames[_video] = len(glob.glob(os.path.join(self.image_dir, _video, '*.jpg')))
        _mask = np.array(Image.open(os.path.join(self.mask_dir, _video, '00000.png')).convert("P"))
        self.num_objects[_video] = np.max(_mask)
        self.shape[_video] = np.shape(_mask)

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
    if self.current_video is None:
      raise IndexError("set a video before enumerating through the dataset")
    if self.is_train:
      return self.bptt_len
    else:
      return self.num_frames[self.current_video]

  def read_proposals(self, raw_proposals, proposal_scores, proposal_categories, video, f, gt_mask):
    proposal_file = os.path.join(self.proposal_dir, video, '{:05d}.pickle'.format(f))
    proposals = pickle.load(open(proposal_file, 'rb'))
    proposals = top_n_predictions_maskrcnn(proposals, self.max_proposals)
    if len(proposals.get_field('mask')) > 0:
      raw_proposals[:len(proposals.get_field('mask'))] = proposals.get_field('mask')[:, 0].data.cpu().numpy()
    else:
      print("WARN: no proposals found in {}".format(proposal_file))
    num_proposals = len(proposals.get_field('mask'))
    proposal_scores[:len(proposals.get_field('mask'))] = proposals.get_field('scores')
    proposal_categories[:len(proposals.get_field('mask'))] = proposals.get_field('labels')

    # remove gt proposal if required
    if self.remove_gt_proposal:# and np.random.choice([True, False], p=[0.3, 0.7]):
      proposal_selected = get_best_match(torch.from_numpy(gt_mask).unsqueeze(0), torch.from_numpy(raw_proposals), num_proposals=torch.tensor(num_proposals, dtype=torch.int),
                                           object_categories=torch.from_numpy(proposal_categories))
      if proposal_selected is not None:
        raw_proposals[proposal_selected[2]] = np.zeros_like(raw_proposals[proposal_selected[2]])
    return num_proposals

  def read_frame(self, oh_proposals, obj_id, shape, video, proposal_scores, proposal_categories, f):
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

    num_proposals = self.read_proposals(oh_proposals, proposal_scores, proposal_categories, video, f, raw_mask)
    raw_masks = raw_mask

    return raw_frames, raw_masks, oh_proposals, num_proposals

  def get_centres(self, proposals):
    c = []
    for proposal in proposals:
      dt = distance_transform_edt(proposal)
      locations = np.where(dt == np.max(dt))
      c+=[[locations[0][0], locations[1][0]]]

    return np.array(c)

  def __getitem__(self, index):
    info = {}
    info['name'] = self.current_video
    info['num_frames'] = self.num_frames[self.current_video]
    num_objects = self.num_objects[self.current_video]
    info['num_objects'] = num_objects
    obj_id = 1
    # shift index to the start index if bptt is used
    index = self.start_index + index

    # retain original shape
    # shape = self.shape[self.current_video] if not (self.is_train and self.MO) else self.crop_size
    shape = self.shape[self.current_video]
    oh_proposals = np.zeros((self.max_proposals,) + self.shape[self.current_video],
                            dtype=np.uint8)
    num_proposals = np.zeros((self.num_frames[self.current_video]))
    proposal_scores = np.zeros(self.max_proposals)
    proposal_categories = np.zeros(self.max_proposals)
    raw_frames, raw_masks, oh_proposals, num_proposals = self.read_frame(oh_proposals, obj_id, shape,
                                                                         self.current_video, proposal_scores,
                                                                         proposal_categories, index)

    # make One-hot channel is object index
    oh_masks = np.zeros(shape + (num_objects,), dtype=np.uint8)
    for o in range(num_objects):
      oh_masks[:, :, o] = (raw_masks == (o + 1)).astype(np.uint8)

    # padding size to be divide by 32
    h, w, _ = oh_masks.shape
    new_h = h + 32 - h % 32
    new_w = w + 32 - w % 32
    # print(new_h, new_w)
    lh, uh = (new_h - h) / 2, (new_h - h) / 2 + (new_h - h) % 2
    lw, uw = (new_w - w) / 2, (new_w - w) / 2 + (new_w - w) % 2
    lh, uh, lw, uw = int(lh), int(uh), int(lw), int(uw)
    pad_masks = np.pad(oh_masks, ((lh, uh), (lw, uw), (0, 0)), mode='constant')
    pad_frames = np.pad(raw_frames, ((lh, uh), (lw, uw), (0, 0)), mode='constant')
    pad_proposals = np.pad(oh_proposals, ((0, 0), (lh, uh), (lw, uw)), mode='constant')
    proposal_centres = self.get_centres(pad_proposals)
    info['pad'] = ((lh, uh), (lw, uw))
    info['num_proposals'] = num_proposals
    info['proposal_scores'] = proposal_scores
    info['proposal_categories'] = proposal_categories

    th_frames = torch.unsqueeze(torch.from_numpy(np.transpose(pad_frames, (2, 0, 1)).copy()).float(), 0)
    th_masks = torch.unsqueeze(torch.from_numpy(np.transpose(pad_masks, (2, 0, 1)).copy()).long(), 0)
    th_proposals = torch.unsqueeze(torch.from_numpy(pad_proposals).long(), 0)
    proposal_centres = torch.from_numpy(proposal_centres).unsqueeze(0).long().permute(0,2,1)

    return th_frames, th_masks, th_proposals, proposal_centres, info
