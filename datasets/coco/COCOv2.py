import os

import numpy as np
from PIL import Image

from datasets.BaseDataset import VideoDataset, INFO, IMAGES_, TARGETS
from datasets.utils.Util import generate_clip_from_image
from utils.Constants import COCO_ROOT
from utils.Resize import ResizeMode


class COCOv2(VideoDataset):
  def __init__(self, root, mode='train', resize_mode=None, resize_shape=None, tw=8, max_temporal_gap=8, num_classes=2,
               restricted_image_category_list = None, exclude_image_category_list = None):
    subset = "train" if mode == "train" else "valid"
    if mode == "train":
      self.data_type = "train2014"
      self.filter_crowd_images = True
      self.min_box_size = 30
    else:
      self.data_type = "val2014"
      self.filter_crowd_images = False
      self.min_box_size = -1.0

    # self.restricted_image_category_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    #                                        'boat', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    #                                        'zebra',
    #                                        'giraffe', 'backpack', 'handbag', 'suitcase', 'frisbee', 'skis', 'snowboard',
    #                                        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    #                                        'surfboard', 'tennis racket', 'remote', 'cell phone']
    self.restricted_image_category_list = restricted_image_category_list
    # Use the minival split as done in https://github.com/rbgirshick/py-faster-rcnn/blob/master/data/README.md
    self.annotation_file = '%s/annotations/instances_%s.json' % (root, subset)
    self.init_coco()
    super(COCOv2, self).__init__(root, mode, resize_mode, resize_shape, tw, max_temporal_gap, num_classes)


  def init_coco(self):
    # only import this dependency on demand
    import pycocotools.coco as coco
    self.coco = coco.COCO(self.annotation_file)
    ann_ids = self.coco.getAnnIds([])
    self.anns = self.coco.loadAnns(ann_ids)
    self.label_map = {k - 1: v for k, v in self.coco.cats.items()}
    self.filename_to_anns = dict()
    self.build_filename_to_anns_dict()

  def build_filename_to_anns_dict(self):
    for ann in self.anns:
      img_id = ann['image_id']
      img = self.coco.loadImgs(img_id)
      file_name = img[0]['file_name']
      if file_name in self.filename_to_anns:
        self.filename_to_anns[file_name].append(ann)
      else:
        self.filename_to_anns[file_name] = [ann]
        # self.filename_to_anns[file_name] = ann
    self.filter_anns()

  def filter_anns(self):
    # exclude all images which contain a crowd
    if self.filter_crowd_images:
      self.filename_to_anns = {f: anns for f, anns in self.filename_to_anns.items()
                               if not any([an["iscrowd"] for an in anns])}
    # filter annotations with too small boxes
    if self.min_box_size != -1.0:
      self.filename_to_anns = {f: [ann for ann in anns if ann["bbox"][2] >= self.min_box_size and ann["bbox"][3]
                                   >= self.min_box_size] for f, anns in self.filename_to_anns.items()}

    # remove annotations with crowd regions
    self.filename_to_anns = {f: [ann for ann in anns if not ann["iscrowd"]]
                             for f, anns in self.filename_to_anns.items()}
    # restrict images to contain considered categories
    if self.restricted_image_category_list is not None:
      print("filtering images to contain categories", self.restricted_image_category_list)
      self.filename_to_anns = {f: anns for f, anns in self.filename_to_anns.items()
                               if any([self.label_map[ann["category_id"] - 1]["name"]
                                       in self.restricted_image_category_list for ann in anns])}
      for cat in self.restricted_image_category_list:
        n_imgs_for_cat = sum([1 for anns in self.filename_to_anns.values() if
                              any([self.label_map[ann["category_id"] - 1]["name"] == cat for ann in anns])])
        print("number of images containing", cat, ":", n_imgs_for_cat)

    # filter out images without annotations
    self.filename_to_anns = {f: anns for f, anns in self.filename_to_anns.items() if len(anns) > 0}
    n_before = len(self.anns)
    self.anns = []
    for anns in self.filename_to_anns.values():
      self.anns += anns
    n_after = len(self.anns)
    print("filtered annotations:", n_before, "->", n_after)

  def generate_clip(self, raw_frame, raw_mask):
    clip_frames, clip_masks = generate_clip_from_image(raw_frame, raw_mask[...,None], self.tw)
    return clip_frames, clip_masks

  def set_video_id(self, video):
    pass

  def get_video_ids(self):
    return [0]

  def create_sample_list(self):
    img_dir = '%s/%s/' % (self.root, self.data_type)

    # Filtering the image file names since some of them do not have annotations.
    # Since we use the minival split, validation files could be part of the training set
    imgs = [os.path.join('%s/%s/' % (self.root,
                                     "train2014" if "train2014" in fn else "val2014"),
                         fn)
            for fn in self.filename_to_anns.keys()]
    for line in imgs:
      _video = line.split("/")[-1].split(".")[0]
      self.videos += [_video]
      self.num_frames[_video] = 1
      sample = {INFO: {}, IMAGES_: [], TARGETS: []}
      sample[IMAGES_] = [line]
      sample[INFO]['video'] = _video
      self.samples+=[sample]

    self.raw_samples = self.samples

  def read_image(self, sample):
    path = sample[IMAGES_][0]
    # path = img_filename.split('/')[-1]
    # img_dir = os.path.join(self.data_dir, "train2014") if path.split('_')[1] == "train2014" else \
    #   os.path.join(self.data_dir, "val2014")
    # path = os.path.join(img_dir, path)
    img = np.array(Image.open(path).convert('RGB'))
    return [img]

  def read_target(self, sample):
    img_filename = sample[IMAGES_][0]
    anns = self.filename_to_anns[img_filename.split("/")[-1]]
    img = self.coco.loadImgs(anns[0]['image_id'])[0]

    height = img['height']
    width = img['width']
    sample[INFO]['shape'] = (height, width)

    label = np.zeros((height, width))
    sample[INFO]['num_objects'] = len(anns)
    for i, ann in enumerate(anns):
      mask = self.coco.annToMask(ann)[:, :]
      label[mask!=0] = i + 1
    if len(np.unique(label)) == 1:
      print("GT contains only background.")

    return [label.astype(np.uint8)]

  def normalise(self, tensors):
    image = tensors[IMAGES_][0]
    mask = tensors[TARGETS][0]
    images, targets = self.generate_clip(image, mask.squeeze())
    tensors[IMAGES_] = images
    tensors[TARGETS] = targets
    return super(COCOv2, self).normalise(tensors)


if __name__ == '__main__':
    davis = COCOv2(root=COCO_ROOT,
                  resize_shape=(480, 854), resize_mode=ResizeMode.FIXED_SIZE, mode="train", max_temporal_gap=32)

    # davis.set_video_id('cat-girl')
    print("Dataset size: {}".format(davis.__len__()))

    for i, _input in enumerate(davis):
      print(_input['info'])
      print("Image Max {}, Image Min {}".format(_input['images'].max(), _input['images'].min()),
            "Target max {}, Target Min {}".format(_input['target']['mask'].max(), _input['target']['mask'].min()))