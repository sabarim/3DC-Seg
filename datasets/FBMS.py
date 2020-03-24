import glob
import os
import random

import numpy as np
from PIL import Image

from datasets.DAVIS import DAVIS
from utils.Resize import ResizeMode, resize


class FBMSDataset(DAVIS):
    def __init__(self, root, is_train=False, crop_size=None, temporal_window=8, min_temporal_gap=2,
                 max_temporal_gap=8, resize_mode=ResizeMode.FIXED_SIZE):
        # maintain a dict to store the index length for videos. They are different for fbms
        self.index_length = {}
        super(FBMSDataset, self).__init__(root=root, is_train=is_train, crop_size=crop_size,
                                          temporal_window=temporal_window,
                                          min_temporal_gap=min_temporal_gap, max_temporal_gap=max_temporal_gap,
                                          resize_mode=resize_mode)

    def set_paths(self, imset, resolution, root):
        subset = "Trainingset" if self.is_train else "Testset"
        self.mask_dir = os.path.join(root, subset, 'GroundTruth')
        self.image_dir = os.path.join(root, subset)
        return self.image_dir

    def set_video_id(self, video):
        self.current_video = video
        self.start_index = self.get_start_index(video)
        self.img_list = list(glob.glob(os.path.join(self.image_dir, self.current_video, '*.jpg')))
        self.img_list.sort()
        # instance_ids = list(range(self.num_objects[video] + 1))
        # instance_ids.remove(0)
        # self.random_instance_ids[video] = random.choice(instance_ids)

    def create_img_list(self, _imset_f):
        videos = glob.glob(self.image_dir + "/*")
        for _video in videos:
            sequence = _video.split("/")[-1]
            self.videos.append(sequence)
            vid_files = glob.glob(os.path.join(self.image_dir, sequence, '*.jpg'))
            self.index_length[sequence] = len(vid_files[0].split("/")[-1].split(".")[0].split("_")[-1])
            self.num_frames[sequence] = len(vid_files)
            self.num_objects[sequence] = 0
            self.shape[sequence] = np.shape(np.array(Image.open(vid_files[0]).convert('RGB')))[:2]
            self.img_list += list(glob.glob(os.path.join(self.image_dir, sequence, '*.jpg')))
        self.reference_list = self.img_list

    def read_frame(self, shape, video, f, instance_id=None, support_indices=None):
        # use a blend of both full random instance as well as the full object
        l = self.index_length[video]
        img_file = os.path.join(self.image_dir, video, video + ('_{:0' + str(l) + 'd}.jpg').format(f))
        raw_frames = np.array(Image.open(img_file).convert('RGB')) / 255.
        raw_mask = np.zeros(raw_frames.shape[:2])
        mask_void = (raw_mask == 255).astype(np.uint8)
        raw_mask[raw_mask == 255] = 0
        raw_masks = (raw_mask == instance_id).astype(np.uint8) if instance_id is not None else raw_mask
        tensors_resized = resize({"image": raw_frames, "mask": raw_masks},
                                 self.resize_mode, shape)
        return tensors_resized["image"] / 255.0, tensors_resized["mask"], mask_void

    def get_support_indices(self, index, sequence):
        # index should be start index of the clip
        if self.is_train:
            index_range = np.arange(index, min(self.num_frames[sequence],
                                               (index + max(self.max_temporal_gap, self.temporal_window))))
        else:
            index_range = np.arange(index,
                                    min(self.num_frames[sequence], (index + self.temporal_window)))

        support_indices = np.random.choice(index_range, min(self.temporal_window, len(index_range)), replace=False)
        support_indices = np.sort(np.append(support_indices, np.repeat([index],
                                                                       self.temporal_window - len(support_indices))))

        # print(support_indices)
        return support_indices

    def __getitem__(self, item):
        img_file = self.img_list[item]
        sequence = self.get_current_sequence(img_file)
        info = {}
        info['name'] = sequence
        info['num_frames'] = self.num_frames[sequence]
        num_objects = self.num_objects[sequence]
        info['num_objects'] = num_objects
        info['shape'] = self.shape[sequence]
        index = int(os.path.splitext(os.path.basename(img_file))[0].split("_")[-1])

        # retain original shape
        # shape = self.shape[self.current_video] if not (self.is_train and self.MO) else self.crop_size
        shape = self.shape[sequence] if self.crop_size is None else self.crop_size
        support_indices = self.get_support_indices(index, sequence)
        info['support_indices'] = support_indices
        th_frames = []
        th_masks = []
        th_mask_void = []
        instance_id = np.random.choice(np.array(range(1, num_objects + 1))) if self.random_instance else None
        # add the current index and the previous frame with respect to the max index in supporting frame
        for i in np.sort(support_indices):
            raw_frame, raw_mask, mask_void = \
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
            pad_frames = np.pad(raw_frame, ((lh, uh), (lw, uw), (0, 0)), mode='constant')
            info['pad'] = ((lh, uh), (lw, uw))

            th_frames.append(np.transpose(pad_frames, (2, 0, 1))[:, np.newaxis])
            th_masks.append(pad_masks[np.newaxis, np.newaxis])
            th_mask_void.append(pad_mask_void[np.newaxis, np.newaxis])
            # th_pc.append(proposal_categories[np.newaxis])
            # th_ps.append(proposal_scores[np.newaxis])

        th_masks_raw = th_masks.copy()
        th_masks[-1] = np.zeros_like(th_masks[-1])
        masks_guidance = np.concatenate(th_masks, axis=1)
        # remove masks with some probability to make sure that the network can focus on intermediate frames
        # if self.is_train and np.random.choice([True, False], 1, p=[0.15,0.85]):
        #   masks_guidance[0, -2] = np.zeros(masks_guidance.shape[2:])

        # Can be used for setting proposals if desired, but for now it isn't neccessary and will be ignored
        return {'images': np.concatenate(th_frames, axis=1),'info': info,
                'target': masks_guidance, "proposals": masks_guidance,
                "raw_proposals": masks_guidance,
                'raw_masks': np.concatenate(th_masks_raw, axis=1),
                'target_extra': {'similarity_ref': masks_guidance,
                                  'similarity_raw_mask': masks_guidance, 'sem_seg':masks_guidance}}


if __name__ == '__main__':
    dataset = FBMSDataset(root="/globalwork/data/fbms/")
    result = dataset.__getitem__(0)
    print("image range: {}\nfgmask: {}\nsupport: {}".format(
        (result['images'].min(),
        result['images'].max()),
        (np.unique(result['target']), result['target'].shape),
        result['info']['support_indices']))