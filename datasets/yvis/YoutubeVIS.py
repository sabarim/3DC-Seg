import copy
import json
import os
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from pycocotools import mask as cocomask

from datasets.BaseDataset import INFO, IMAGES_, TARGETS
from datasets.yvos.YoutubeVOS import YoutubeVOS
from utils.Constants import YOUTUBEVIS_ROOT
from utils.Resize import ResizeMode


class YoutubeVISDataset(YoutubeVOS):
  def __init__(self, root, mode='train', resize_mode=None, resize_shape=None, tw=8, max_temporal_gap=8, num_classes=2):
    self.instances = {}
    super(YoutubeVISDataset, self).__init__(root, mode, resize_mode, resize_shape, tw, max_temporal_gap, num_classes)

  def create_sample_list(self):
    start = time.time()
    json_set = 'train' if self.is_train() else 'valid'
    mask_dir = os.path.join(self.root, 'VIS-labels', json_set + '.json')
    with open(mask_dir, 'r') as readfile:
      json_content = json.load(readfile)
    anns = defaultdict(list)
    if 'annotations' in json_content:
      for ann in json_content['annotations']:
        anns[ann['video_id']].append(ann)

    _ = [self.video_list(video, anns) for video in json_content['videos']]
    self.raw_samples = self.samples
    print("Time to create image list".format(time.time() - start))

  def video_list(self, _video, anns):
    start_time = time.time()
    imset = 'train' if self.is_train() else 'valid-VIS'
    image_dir = os.path.join(self.root, 'images/', imset, 'JPEGImages')
    img_list = [os.path.join(image_dir, filename) for filename in _video['file_names']]

    video_name = img_list[0].split('/')[-2]
    self.video_frames[video_name] = [int(os.path.splitext(os.path.basename(f))[0]) for f in img_list]
    shape = (_video['height'], _video['width'])
    if anns is not None:
      instances = [ann for ann in anns[_video['id']]]
      n_objects = len(instances)
    else:
      n_objects = 0
      instances = []
    num_frames = len(img_list)

    for f_index in self.video_frames[video_name]:
      sample = {INFO: {}, IMAGES_: [], TARGETS: []}
      support_indices = self.get_support_indices(f_index, video_name)
      sample[INFO]['support_indices'] = support_indices

      images = [os.path.join(image_dir, video_name, '{:05d}.jpg'.format(s)) for s in np.sort(support_indices)]

      _ids = [self.video_frames[video_name].index(s) for s in np.sort(support_indices)]
      filtered_instances = {}
      for i, instance in enumerate(instances):
        filtered_instances[i] = {}
        filtered_instances[i]['segmentations'] = np.array(instance['segmentations'])[_ids]
        filtered_instances[i]['bboxes'] = np.array(instance['bboxes'])[_ids]
        filtered_instances[i]['areas'] = np.array(instance['areas'])[_ids]
        filtered_instances[i]['category_id'] = instance['category_id']
      targets = filtered_instances

      sample[IMAGES_] = images
      sample[TARGETS] = targets

      sample[INFO]['video'] = video_name
      sample[INFO]['num_frames'] = num_frames
      sample[INFO]['num_objects'] = n_objects
      sample[INFO]['shape'] = shape

      self.samples += [sample]

    print("Time taken for vide {}: {}".format(video_name, time.time()-start_time))
    return video_name, img_list, n_objects, num_frames, shape, instances

  def read_target(self, sample):
    def get_mask(_idx):
      mask = np.zeros(sample[INFO]['shape'])
      for obj_id, _instance in sample[TARGETS].items():
        if _instance['segmentations'][obj_id] is not None:
          _m = cocomask.decode(cocomask.frPyObjects(_instance['segmentations'][obj_id],
                                                              *_instance['segmentations'][obj_id]['size']))
          mask[_m != 1] = obj_id + 1
      return mask.astype(np.uint8)

    masks = [get_mask(_idx) for _idx in range(self.tw)]


                       # for instance in instances if instance['segmentations'][idx] is not None]
    return masks


if __name__ == '__main__':
    j = os.path.join(YOUTUBEVIS_ROOT, 'VIS-labels', 'valid.json')
    j_c = json.load(open(j))
    print([c['name'] for c in j_c['categories']])
    yvos = YoutubeVISDataset(root=YOUTUBEVIS_ROOT,
                      resize_shape=(480, 854), resize_mode=ResizeMode.FIXED_SIZE, mode="train", max_temporal_gap=8)

    # davis.set_video_id('cat-girl')
    print("Dataset size: {}".format(yvos.__len__()))

    for i, _input in enumerate(yvos):
      print(_input['info'])
      print("Image Max {}, Image Min {}".format(_input['images'].max(), _input['images'].min()),
            "Target max {}, Target Min {}".format(_input['target']['mask'].max(), _input['target']['mask'].min()))