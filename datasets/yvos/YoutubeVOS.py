import glob
import os

import numpy as np
from PIL import Image

from datasets.BaseDataset import VideoDataset, IMAGES_, TARGETS, INFO
from utils.Constants import YOUTUBEVOS_ROOT
from utils.Resize import ResizeMode


class YoutubeVOS(VideoDataset):
  def __init__(self, root, mode='train', resize_mode=None, resize_shape=None, tw=8, max_temporal_gap=8, num_classes=2):
    self.videos = []
    self.num_frames = {}
    self.num_objects = {}
    self.shape = {}
    self.raw_samples = []
    self.video_frames = {}
    super(YoutubeVOS, self).__init__(root, mode, resize_mode, resize_shape, tw, max_temporal_gap, num_classes)

  def filter_samples(self, video):
    filtered_samples = [s for s in self.raw_samples if s[INFO]['video'] == video]
    self.samples = filtered_samples

  def get_support_indices(self, index, sequence):
    # in youtube-vos index does not correspond to the file index
    # i = int(os.path.splitext(os.path.basename(self.img_list[file_index]))[0])
    # sample_list = [int(os.path.splitext(os.path.basename(f))[0]) for f in self.video_frames[sequence]]
    sample_list = self.video_frames[sequence]
    sample_list.sort()
    start_index = sample_list.index(index)
    end_index = min(len(sample_list), start_index + self.max_temporal_gap)
    sample_list = sample_list[start_index: end_index]
    support_indices = np.random.choice(sample_list, min(self.tw, len(sample_list)), replace=False)
    support_indices = np.sort(np.append(support_indices, np.repeat([index],
                                                                   self.tw - len(support_indices))))
    support_indices.sort()
    # print("support indices are {}".format(support_indices))
    return support_indices.astype(np.int)

  def create_sample_list(self):
    imset = "train" if self.is_train() else "valid"
    image_dir = os.path.join(self.root, imset, 'JPEGImages')
    _videos = glob.glob(image_dir + "/*")

    for line in _videos:
      _video = line.split("/")[-1]
      self.videos += [_video]
      img_list = list(glob.glob(os.path.join(image_dir, _video, '*.jpg')))
      mask_dir = os.path.join(self.root, imset, 'CleanedAnnotations')
      if os.path.exists(os.path.join(mask_dir, _video)):
        mask_list = list(glob.glob(os.path.join(mask_dir, _video, '*.png')))
      else:
        mask_dir = os.path.join(self.root, imset, 'Annotations')
        mask_list = list(glob.glob(os.path.join(mask_dir.replace("CleanedAnnotations", "Annotations"),
                                                _video, '*.png')))
      img_list.sort()
      mask_list.sort()
      self.video_frames[_video] = [int(os.path.splitext(os.path.basename(f))[0]) for f in img_list]

      num_frames = len(img_list)
      self.num_frames[_video] = num_frames

      _mask = np.shape(np.array(Image.open(mask_list[0]).convert("P")))
      num_objects = np.max(_mask)
      self.num_objects[_video] = num_objects
      self.shape[_video] = np.shape(_mask)

      for f_index in self.video_frames[_video]:
        sample = {INFO: {}, IMAGES_: [], TARGETS: []}
        support_indices = self.get_support_indices(f_index, _video)
        sample[INFO]['support_indices'] = support_indices
        images = [os.path.join(image_dir, _video, '{:05d}.jpg'.format(s)) for s in np.sort(support_indices)]
        targets = [os.path.join(mask_dir, _video, '{:05d}.png'.format(s)) for s in np.sort(support_indices)]
        sample[IMAGES_] = images
        sample[TARGETS] = targets

        sample[INFO]['video'] = _video
        sample[INFO]['num_frames'] = num_frames
        sample[INFO]['num_objects'] = num_objects
        sample[INFO]['shape'] = np.shape(_mask)

        self.samples+=[sample]
    self.raw_samples = self.samples


if __name__ == '__main__':
    yvos = YoutubeVOS(root=YOUTUBEVOS_ROOT,
                  resize_shape=(480, 854), resize_mode=ResizeMode.FIXED_SIZE, mode="train", max_temporal_gap=32)

    # davis.set_video_id('cat-girl')
    print("Dataset size: {}".format(yvos.__len__()))

    for i, _input in enumerate(yvos):
      print(_input['info'])
      print("Image Max {}, Image Min {}".format(_input['images'].max(), _input['images'].min()),
            "Target max {}, Target Min {}".format(_input['target']['mask'].max(), _input['target']['mask'].min()))