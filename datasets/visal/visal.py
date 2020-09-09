import glob
import os
import re

import numpy as np
from PIL import Image
from imageio import imread

from datasets.BaseDataset import INFO, IMAGES_, TARGETS
from datasets.davis.Davis import Davis
from utils.Resize import ResizeMode

SEQ_NAMES = [
    "aeroplane", "bird", "boat", "boat2", "car", "cat", "cow4", "cow5", "gokart", "horse2",
    "horse3", "lion", "man", "motorbike2", "panda", "rider", "snow_leopards"
]


class VisalDataset(Davis):
    def __init__(self, root, mode='train', resize_mode=None, resize_shape=None, tw=8, max_temporal_gap=8, num_classes=2,
                 imset=None):
        self.gt_frames = {}
        self.video_frames = {}
        super(VisalDataset, self).__init__(root, mode, resize_mode, resize_shape, tw, max_temporal_gap, num_classes)

    def get_current_sequence(self, img_file):
        sequence = img_file.split("/")[-2]
        return sequence

    def get_support_indices(self, index, sequence):
        # index should be start index of the clip
        if self.is_train():
            index_range = np.arange(index, min(self.num_frames[sequence],
                                               (index + max(self.max_temporal_gap, self.tw))))
        else:
            index_range = np.arange(index,
                                    min(self.num_frames[sequence], (index + self.tw)))

        support_indices = np.random.choice(index_range, min(self.tw, len(index_range)), replace=False)
        support_indices = np.sort(np.append(support_indices, np.repeat([index],
                                                                       self.tw - len(support_indices))))

        return support_indices

    def read_target(self, sample):
        masks = []
        for t in sample[TARGETS]:
            if os.path.exists(t):
                raw_mask = np.array(Image.open(t).convert('P'), dtype=np.uint8)
                raw_mask = (raw_mask!=0).astype(np.uint8)
                mask_void = (raw_mask == 255).astype(np.uint8)
                raw_mask[raw_mask == 255] = 0
            else:
                raw_mask = np.zeros(sample[INFO]['shape']).astype(np.uint8)
                mask_void = (np.ones_like(raw_mask) * 255).astype(np.uint8)
            masks +=[raw_mask]
        return masks

    def create_sample_list(self):
        image_dir = os.path.join(self.root, "ViSal")
        mask_dir = os.path.join(self.root, "GroundTruth")

        assert os.path.exists(image_dir), "Images directory not found at expected path: {}".format(
            image_dir)
        assert os.path.exists(mask_dir), "Ground truth directory not found at expected path: {}".format(
            mask_dir)
        types = ('/*.bmp', '/*.png')
        mask_fnames = []
        for type in types:
            mask_fnames += sorted(glob.glob(mask_dir + type))
        mask_fnames = [fname.split("/")[-1] for fname in mask_fnames]
        for _video in SEQ_NAMES:
            self.videos.append(_video)
            seq_images_dir = os.path.join(image_dir, _video)
            assert os.path.exists(seq_images_dir), "Images directory not found at expected path: {}".format(
                seq_images_dir)
            print("Reading Sequence {}".format(_video))
            if _video in ("gokart", "snow_leopards"):
                regex_pattern = _video + r"[^a-zA-Z]"
            else:
                regex_pattern = _video + r"[^a-zA-Z0-9]"

            vid_files = []
            for type in types:
                vid_files += sorted(glob.glob(seq_images_dir + type))
            seq_mask_fnames = sorted(filter(lambda f: re.match(regex_pattern, f), mask_fnames))
            self.gt_frames[_video] = [int(i) for i, f in enumerate(vid_files) if f.split("/")[-1] in seq_mask_fnames]
            assert len(self.gt_frames[_video]) == len(seq_mask_fnames)
            self.num_frames[_video] = len(vid_files)
            self.video_frames[_video] = vid_files
            shape = imread(vid_files[0]).shape[:2]

            for i, _f in enumerate(vid_files):
                sample = {INFO: {}, IMAGES_: [], TARGETS: []}
                sequence = self.get_current_sequence(_f)
                index = self.video_frames[sequence].index(_f)
                support_indices = self.get_support_indices(index, _video)

                sample[INFO]['support_indices'] = support_indices
                images = [self.video_frames[_video][s] for s in np.sort(support_indices)]
                targets = [os.path.join(mask_dir, img_file.split("/")[-1]) for img_file in images]

                sample[IMAGES_] = images
                sample[TARGETS] = targets
                sample[INFO]['video'] = _video
                sample[INFO]['num_frames'] = len(vid_files)
                sample[INFO]['num_objects'] = 1
                sample[INFO]['shape'] = shape
                sample[INFO]['gt_frames'] = self.gt_frames[_video]

                self.samples += [sample]

        self.raw_samples = self.samples


if __name__ == '__main__':
    davis = VisalDataset(root="/globalwork/mahadevan/mywork/data/ViSal/",
                  resize_shape=(480, 854), resize_mode=ResizeMode.FIXED_SIZE, mode="train", max_temporal_gap=8)

    # davis.set_video_id('cat-girl')
    print("Dataset size: {}".format(davis.__len__()))

    for i, _input in enumerate(davis):
      print(_input['info'])
      print("Image Max {}, Image Min {}".format(_input['images'].max(), _input['images'].min()),
            "Target max {}, Target Min {}".format(_input['target']['mask'].max(), _input['target']['mask'].min()))
