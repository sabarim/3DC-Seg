import glob
import os
import random
from PIL import Image
from torch.utils import data
import torch
import numpy as np
import pycocotools.mask as cocomask

from datasets.DAVIS import DAVIS
from datasets.DAVIS16 import DAVIS16
from datasets.coco.COCO import COCO_SUPERCATEGORIES
from datasets.utils.Util import generate_clip_from_image
from utils.Constants import KITTI_ROOT
from utils.Resize import ResizeMode, resize

SEQ_IDS_TRAIN = ["%04d" % idx for idx in [0, 1, 3, 4, 5, 9, 11, 12, 15, 17, 19, 20]]
SEQ_IDS_VAL = ["%04d" % idx for idx in [2, 6, 7, 8, 10, 13, 14, 16, 18]]
VOID_CLASS = 10
TIMESTEPS_PER_SEQ = {"0000": 154, "0001": 447, "0002": 233, "0003": 144, "0004": 314, "0005": 297, "0006": 270,
                     "0007": 800, "0008": 390, "0009": 803, "0010": 294, "0011": 373, "0012": 78, "0013": 340,
                     "0014": 106, "0015": 376, "0016": 209, "0017": 145, "0018": 339, "0019": 1059, "0020": 837}



class KITTIDataset(DAVIS16):
  def __init__(self, root, is_train=False,crop_size=None, temporal_window=8, max_temporal_gap=8,
               resize_mode=ResizeMode.FIXED_SIZE, min_size = 0):
    if is_train:
      self.sequences = SEQ_IDS_TRAIN
    else:
      self.sequences = SEQ_IDS_VAL
    self.validation_set_size = -1
    self.is_train = is_train
    self.min_size = min_size
    self._id_divisor = 1000
    self.coco_cats_mapping = {1:COCO_SUPERCATEGORIES.index('vehicle') + 1, 2: COCO_SUPERCATEGORIES.index('person') + 1,
                              10: 10}
    super(KITTIDataset, self).__init__(root, is_train=is_train, num_classes=2, crop_size=crop_size,
                                       temporal_window=temporal_window, max_temporal_gap=max_temporal_gap, resize_mode=resize_mode)
  def set_paths(self, imset, resolution, root):
    self.mask_dir = os.path.join(root, 'train','instances')
    self.image_dir = os.path.join(root, 'train', 'images')
    return self.sequences

  def create_img_list(self, _imset_f):
    for line in _imset_f:
      _video = line.rstrip('\n')
      self.videos.append(_video)
      self.num_frames[_video] = len(glob.glob(os.path.join(self.image_dir, _video, '*.png')))
      _mask = np.array(Image.open(os.path.join(self.mask_dir, _video, '000000.png')).convert("P"))
      self.num_objects[_video] = np.max(_mask)
      self.shape[_video] = np.shape(_mask)
      self.img_list += list(glob.glob(os.path.join(self.image_dir, _video, '*.png')))

    if self.is_train:
      self.reference_list = [f for f in self.img_list if np.sum(
        np.array(Image.open(f.replace('images', 'instances'))) != 10000)]


  def set_video_id(self, video):
    self.current_video = video
    self.start_index = self.get_start_index(video)
    self.img_list = list(glob.glob(os.path.join(self.image_dir, self.current_video, '*.png')))
    self.img_list.sort()
    instance_ids = list(range(self.num_objects[video] + 1))
    instance_ids.remove(0)
    self.random_instance_ids[video] = random.choice(instance_ids)

  def __len__(self):
    return len(self.reference_list) if self.is_train else len(self.img_list)

  def get_instance_masks(self, raw_mask):
    sem_seg = (raw_mask // self._id_divisor).astype(np.uint8)
    masks_instances = (raw_mask % self._id_divisor).astype(np.uint8)
    masks_instances[sem_seg == VOID_CLASS] = 0
    void_mask = (sem_seg != VOID_CLASS).astype(np.uint8)
    sem_seg[sem_seg == VOID_CLASS] = 0
    # remap to coco supercategories
    for key in self.coco_cats_mapping:
      sem_seg[sem_seg == key] = self.coco_cats_mapping[key]

    return masks_instances, sem_seg, void_mask

  def read_frame(self, shape, video, f, instance_id=None, support_indices=None):
    tensor = {'image': [], 'target':[], 'instance':[], 'sem_seg':[], 'void_label': []}
    # use a blend of both full random instance as well as the full object
    for f in support_indices:
      img_file = os.path.join(self.image_dir, video, '{:06d}.png'.format(f))
      raw_frame = np.array(Image.open(img_file).convert('RGB')) / 255.
      mask_file = os.path.join(self.mask_dir, video, '{:06d}.png'.format(f))  # always return first frame mask
      raw_mask = np.array(Image.open(mask_file), dtype=np.uint16)
      mask_instances, sem_seg, void_label = self.get_instance_masks(raw_mask)

      tensors_resized = resize({"image": raw_frame, "mask": mask_instances, "sem_seg":sem_seg, "void_label": void_label},
                               self.resize_mode, self.crop_size)
      mask_instances = tensors_resized['mask']

      # select object
      # ids = np.unique(mask_instances)
      # ids = [id_ for id_ in ids if id_ // self._id_divisor in (1, 2)]

      # generate semantic segmentation mask using category ids
      # masks_cat_ids = np.zeros_like(mask_instances)
      # mask_void = np.ones_like(mask_instances)
      # for i, cat in enumerate(cats):
      #   if cat in (self.coco_cats_mapping[1], self.coco_cats_mapping[2]):
      #     masks_cat_ids[mask_instances == i + 1] = cat
      #   else:
      #     mask_void[mask_instances == i + 1] = 0
      #     remove void label instances
      #     mask_instances[mask_instances == i + 1] = 0


      tensor['image'] += [tensors_resized['image'] / 255.]
      tensor['target'] += [(mask_instances != 0)[..., None].astype(np.uint8)]
      tensor['instance'] += [mask_instances[..., None].astype(np.uint8)]
      tensor['sem_seg'] += [tensors_resized['sem_seg'][..., None].astype(np.uint8)]
      tensor['void_label'] += [tensors_resized['void_label'][..., None].astype(np.uint8)]

    tensors = [(key, np.transpose(np.stack(val), axes=(3, 0, 1, 2))) for key, val in tensor.items()]

    return dict(tensors)

  def __getitem__(self, index):
    img_file = self.reference_list[index] if self.is_train else self.img_list[index]
    sequence = self.get_current_sequence(img_file)
    info = {}
    info['name'] = sequence
    info['num_frames'] = self.num_frames[sequence]
    num_objects = self.num_objects[sequence]
    info['num_objects'] = num_objects
    info['shape'] = self.shape[sequence]
    obj_id = 1
    index = int(os.path.splitext(os.path.basename(img_file))[0])

    shape = self.shape[sequence] if self.crop_size is None else self.crop_size
    support_indices = self.get_support_indices(index, sequence)
    info['support_indices'] = support_indices
    tensor = self.read_frame(shape, sequence, None, None, support_indices)
    masks_instances = tensor['instance']

    info['num_objects'] = len(np.unique(masks_instances))
    # padding size to be divide by 32
    _, _, h, w = masks_instances.shape
    new_h = h + 32 - h % 32 if h % 32 > 0 else h
    new_w = w + 32 - w % 32 if w % 32 > 0 else w
    # print(new_h, new_w)
    lh, uh = (new_h - h) / 2, (new_h - h) / 2 + (new_h - h) % 2
    lw, uw = (new_w - w) / 2, (new_w - w) / 2 + (new_w - w) % 2
    lh, uh, lw, uw = int(lh), int(uh), int(lw), int(uw)
    pad_masks = np.pad(tensor['target'], ((0, 0), (0, 0), (lh, uh), (lw, uw)), mode='constant')
    pad_masks_sem_seg = np.pad(tensor['sem_seg'], ((0, 0), (0, 0), (lh, uh), (lw, uw)), mode='constant')
    pad_masks_instances = np.pad(masks_instances, ((0, 0), (0, 0), (lh, uh), (lw, uw)), mode='constant')
    pad_frames = np.pad(tensor['image'], ((0, 0), (0, 0), (lh, uh), (lw, uw)), mode='constant')
    info['pad'] = ((lh, uh), (lw, uw))

    return {'images': pad_frames, 'info': info,
            'target': pad_masks, "proposals": pad_masks, "raw_proposals": pad_masks,
            'raw_masks': pad_masks,
            'target_extra': {'sem_seg': pad_masks_sem_seg,
                             'similarity_raw_mask': pad_masks_instances}}


if __name__ == '__main__':
  dataset = KITTIDataset(KITTI_ROOT, is_train=False, crop_size=[256, 448])
  dataset.set_video_id(dataset.get_video_ids()[0])
  result = dataset.__getitem__(200)
  print(dataset.__len__())
  print("cats: {}\n instances: {}\n image range: {}\nfgmask: {}".format(np.unique(result['target_extra']['sem_seg']),
                                                                        np.unique(result['target_extra'][
                                                                                    'similarity_raw_mask']),
                                                                        (
                                                                        result['images'].min(), result['images'].max()),
                                                                        np.unique(result['raw_masks'])))