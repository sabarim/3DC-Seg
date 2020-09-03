import glob
import os

import numpy as np
from PIL import Image
from imageio import imread

from datasets.BaseDataset import INFO, IMAGES_, TARGETS
from datasets.davis.Davis import Davis
from utils.Constants import FBMS_ROOT
from utils.Resize import ResizeMode


class FBMSDataset(Davis):
  def __init__(self, root, mode='train', resize_mode=None, resize_shape=None, tw=8, max_temporal_gap=8, num_classes=2):
    # maintain a dict to store the index length for videos. They are different for fbms
    self.index_length = {}
    self.gt_frames = {}
    self.video_frames = {}
    super(FBMSDataset, self).__init__(root, mode, resize_mode, resize_shape, tw, max_temporal_gap, num_classes)

  def get_support_indices(self, index, sequence):
    # index should be start index of the clip
    if self.is_train():
      index_range = np.arange(index, min(self.num_frames[sequence],
                                         (index + max(self.max_temporal_gap, self.tw))))
    else:
      index_range = np.arange(index,
                              min(max(self.video_frames[sequence]) + 1, (index + self.tw)))

    support_indices = np.random.choice(index_range, min(self.tw, len(index_range)), replace=False)
    support_indices = np.sort(np.append(support_indices, np.repeat([index],
                                                                   self.tw - len(support_indices))))

    # print(support_indices)
    return support_indices

  def read_target(self, sample):
    masks = []
    for t in sample[TARGETS]:
      if os.path.exists(t):
        raw_mask = np.array(Image.open(t).convert('P'), dtype=np.uint8)
        raw_mask = (raw_mask != 0).astype(np.uint8)
        mask_void = (raw_mask == 255).astype(np.uint8)
        raw_mask[raw_mask == 255] = 0
      else:
        raw_mask = np.zeros(sample[INFO]['shape']).astype(np.uint8)
        mask_void = (np.ones_like(raw_mask) * 255).astype(np.uint8)
      masks += [raw_mask]

    return masks

  def create_sample_list(self):
    subset = "train" if self.is_train() else "test"
    mask_dir = os.path.join(self.root, 'inst', subset)
    subset = "Trainingset" if self.is_train() else "Testset"
    image_dir = os.path.join(self.root, subset)

    videos = glob.glob(image_dir + "/*")
    for _video in videos:
      sequence = _video.split("/")[-1]
      self.videos.append(sequence)
      vid_files = glob.glob(os.path.join(image_dir, sequence, '*.jpg'))
      shape = imread(vid_files[0]).shape[:2]
      self.index_length[sequence] = len(vid_files[0].split("/")[-1].split(".")[0].split("_")[-1])
      self.gt_frames[sequence] = [int(f.split("/")[-1].split("_")[-1].split(".")[0])
                                  for f in glob.glob(os.path.join(mask_dir, sequence, '*.png'))]
      self.num_frames[sequence] = len(vid_files)
      self.video_frames[sequence] = [int(f.split("/")[-1].split("_")[-1].split(".")[0])
                                     for f in vid_files]

      for _f in vid_files:
        sample = {INFO: {}, IMAGES_: [], TARGETS: []}
        index = int(os.path.splitext(os.path.basename(_f))[0].split("_")[-1])
        support_indices = self.get_support_indices(index, sequence)
        sample[INFO]['support_indices'] = support_indices
        l = self.index_length[sequence]
        images = [os.path.join(image_dir, sequence, sequence + ('_{:0' + str(l) + 'd}.jpg').format(s))
                  for s in np.sort(support_indices)]
        targets = [os.path.join(mask_dir, sequence, sequence + ('_{:0' + str(l) + 'd}.png').format(s))
                     for s in np.sort(support_indices)]

        # images = [os.path.join(image_dir, _video, '{:05d}.jpg'.format(s)) for s in np.sort(support_indices)]
        # targets = [os.path.join(mask_dir, _video, '{:05d}.png'.format(s)) for s in np.sort(support_indices)]
        sample[IMAGES_] = images
        sample[TARGETS] = targets

        sample[INFO]['video'] = sequence
        sample[INFO]['num_frames'] = len(vid_files)
        sample[INFO]['num_objects'] = 1
        sample[INFO]['shape'] = shape
        sample[INFO]['gt_frames'] = self.gt_frames[sequence]

        self.samples += [sample]

      self.raw_samples = self.samples


if __name__ == '__main__':
  fbms = FBMSDataset(root=FBMS_ROOT,
                     resize_shape=(480, 854), resize_mode=ResizeMode.FIXED_SIZE, mode="test")

  # fbms.set_video_id('marple4')
  print("Dataset size: {}".format(fbms.__len__()))

  for i, _input in enumerate(fbms):
    print(_input['info'])
    print("Image Max {}, Image Min {}".format(_input['images'].max(), _input['images'].min()),
          "Target max {}, Target Min {}".format(_input['target']['mask'].max(), _input['target']['mask'].min()))
