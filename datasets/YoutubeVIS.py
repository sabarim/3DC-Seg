import glob
import json
import os
import time
import numpy as np
from PIL import Image
from collections import defaultdict
from pycocotools import mask as cocomask
from datasets.YoutubeVOS import YoutubeVOSDataset
from util import get_one_hot_vectors
from utils.Constants import YOUTUBEVIS_ROOT
from utils.Resize import ResizeMode, resize


class YoutubeVISDataset(YoutubeVOSDataset):
  def __init__(self, root, imset='train', is_train=False,
               random_instance=False, crop_size=None,temporal_window=3, min_temporal_gap=2,
               max_temporal_gap=8, resize_mode=ResizeMode.FIXED_SIZE, num_classes=40):
    self.instances = {}
    super(YoutubeVISDataset, self).__init__(root, imset, is_train, random_instance, crop_size, temporal_window,
                                            min_temporal_gap, max_temporal_gap, resize_mode, num_classes)

  def set_paths(self, imset, resolution, root):
    imset = 'train' if self.is_train else 'valid-VIS'
    json_set = 'train' if self.is_train else 'valid'
    self.mask_dir = os.path.join(root, 'VIS-labels', json_set + '.json')
    self.image_dir = os.path.join(root, 'images/', imset, 'JPEGImages')
    # _imset_dir = os.path.join('images/', root, imset)
    # _imset_f = glob.glob(_imset_dir + "/JPEGImages/*")
    return None

  def create_img_list(self, _imset_f):
    start = time.time()
    with open(self.mask_dir, 'r') as readfile:
      json_content = json.load(readfile)
    anns = defaultdict(list)
    for ann in json_content['annotations']:
      anns[ann['video_id']].append(ann)

    results = [self.video_list(video, anns) for video in json_content['videos']]
    results = np.array(results)
    self.videos = results[:, 0]
    self.img_list = np.concatenate(results[:, 1])
    self.reference_list = self.img_list

    video_frames = np.array([results[:, 0], results[:, 1]]).transpose()
    self.video_frames.update({video_frames[i][0]: video_frames[i][1] for i in range(len(video_frames))})

    instances = np.array([results[:, 0], results[:, -1]]).transpose()
    self.instances.update({instances[i][0]: instances[i][1] for i in range(len(instances))})

    shapes = np.array([results[:, 0], results[:, -2]]).transpose()
    self.shape.update({shapes[i][0]: shapes[i][1] for i in range(len(shapes))})
    num_objects = np.array([results[:, 0], results[:, 2]]).transpose()
    self.num_objects.update({num_objects[i][0]: num_objects[i][1] for i in range(len(num_objects))})
    num_frames = np.array([results[:, 0], results[:, -2]]).transpose()
    self.num_frames.update({num_frames[i][0]: num_frames[i][1] for i in range(len(num_frames))})

    print("Time to create image list".format(time.time() - start))

  def video_list(self, video, anns):
    start = time.time()
    # video_content = anns[video['id']]
    img_list = [os.path.join(self.image_dir, filename) for filename in video['file_names']]
    video_name = img_list[0].split('/')[-2]
    shape = (video['height'], video['width'])
    instances = [ann for ann in anns[video['id']]]
    n_objects = len(instances)
    num_frames = len(img_list)

    print("Time to create image list for video {} is {}".format(video_name, time.time() - start))
    return video_name, img_list, n_objects, num_frames, shape, instances

  def read_frame(self, shape, video, f, instance_id=None, support_indices=None):
    # use a blend of both full random instance as well as the full object
    img_file = os.path.join(self.image_dir, video, '{:05d}.jpg'.format(f))
    raw_frames = np.array(Image.open(img_file).convert('RGB')) / 255.
    instances = self.instances[video]
    idx = self.video_frames[video].index(img_file)

    segmentations = [cocomask.decode(cocomask.frPyObjects(instance['segmentations'][idx],
                                                          *instance['segmentations'][idx]['size']))*instance['category_id']
                     for instance in instances if instance['segmentations'][idx] is not None]
    if len(segmentations) > 0:
      sem_seg = np.sum(segmentations, axis=0).astype(np.uint8)
      raw_mask = np.sum([(seg!=0)*(n+1) for n, seg in enumerate(segmentations)], axis=0).astype(np.uint8)
    else:
      sem_seg = np.zeros(self.shape[video]).astype(np.uint8)
      raw_mask = np.zeros(self.shape[video]).astype(np.uint8)

    mask_void = (raw_mask == 255).astype(np.uint8)
    tensors_resized = resize({"image": raw_frames, "mask": raw_mask, 'sem_seg': sem_seg}, self.resize_mode, shape)

    return tensors_resized["image"] / 255.0, tensors_resized["mask"], tensors_resized["sem_seg"], \
           tensors_resized["sem_seg"], mask_void

  def __getitem__(self, item):
    input_dict = super(YoutubeVISDataset, self).__getitem__(item)
    sem_seg = input_dict['raw_proposals']
    input_dict['target_extra'] = {'sem_seg': sem_seg,
                                  'similarity_raw_mask': input_dict['target']}
    input_dict['target'] = (input_dict['target'] != 0).astype(np.uint8)\

    return input_dict



if __name__ == '__main__':
    dataset = YoutubeVISDataset(YOUTUBEVIS_ROOT, is_train=True)
    dataset.__getitem__(110)