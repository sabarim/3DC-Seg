import numpy as np

from datasets.coco.COCO import COCO_SUPERCATEGORIES
from datasets.mapillary.MapillaryBase import MapillaryBaseDataset
from datasets.utils.Util import generate_clip_from_image
from utils.Constants import MAPILLARY_ROOT
from utils.Resize import ResizeMode, resize

NAME = "mapillary_instance"


class MapillaryVideoDataset(MapillaryBaseDataset):
  def __init__(self, root, is_train=False, crop_size=None,temporal_window=8, resize_mode=ResizeMode.FIXED_SIZE,
               resolution = 'quarter', min_size = 0):
    assert resolution in ("quarter", "half", "full"), resolution
    if resolution == "full":
      default_path = root
    else:
      default_path = root.replace("/mapillary/", "/mapillary_{}/".format(resolution))
    self.temporal_window = temporal_window

    # there are 37 classes with instances in total

    # we excluded the following:
    #  8: construction--flat--crosswalk-plain -> doesn't really look like a useful object category
    # 34: object--bike-rack -> holes*
    # 45: object--support--pole -> very large and thin -> bounding box does not capture it well
    # 46: object--support--traffic-sign-frame -> holes*
    # 47: object--support--utility-pole -> holes*

    # further candidate for exclusion:
    #  0: animal--bird  -> usually very small

    # *: holes means that there are large "holes" in the object which usually are still annotated as part of the object
    # this will not work well together with laser, so we exclude them
    vehicle_ids = [52, 53, 54, 55, 56, 57, 59, 60, 61, 62]
    human_ids = [19, 20, 21, 22]
    animal_ids = [0, 1]
    object_ids = [32, 33, 35, 36, 37, 38, 39, 40, 41, 42, 44, 48, 49, 50, 51]
    crosswalk_zebra_id = [23]
    cat_ids_to_use = vehicle_ids + human_ids + animal_ids

    vehicle_coco_mapping = COCO_SUPERCATEGORIES.index('vehicle') + 1
    human_coco_mapping = COCO_SUPERCATEGORIES.index('person') + 1
    animal_coco_mapping = COCO_SUPERCATEGORIES.index('animal') + 1
    self.coco_cats_mapping = {}
    self.coco_cats_mapping.update({key: vehicle_coco_mapping for key in vehicle_ids})
    self.coco_cats_mapping.update({key: human_coco_mapping for key in human_ids})
    self.coco_cats_mapping.update({key: animal_coco_mapping for key in animal_ids})

    super().__init__(default_path, is_train, 256, cat_ids_to_use, crop_size=crop_size, resize_mode=resize_mode,
                     min_size=min_size, name="datasets/Mapillary/")


  def generate_clip(self, raw_frame, raw_mask):
    clip_frames, clip_masks = generate_clip_from_image(raw_frame, raw_mask, self.temporal_window)
    # clip_frames = np.repeat(raw_frame[None], self.temporal_window, axis=0)
    # clip_masks = np.repeat(raw_mask[None], self.temporal_window, axis=0)
    return (clip_frames.astype(np.float32) / 255.0).astype(np.float32), clip_masks

  def get_instance_masks(self, raw_mask):
    ids = np.setdiff1d(np.unique(raw_mask), [0])
    cats = ids // self._id_divisor
    # map category ids to coco supercategories
    cats = np.array([self.coco_cats_mapping[cat] for cat in cats])
    masks_instances = np.zeros_like(raw_mask)
    for i in range(len(ids)):
      masks_instances[raw_mask == ids[i]] = i + 1

    return masks_instances.astype(np.uint8), cats

  def process_input(self, item):
    raw_frame = self.load_image(self.imgs[item])
    # raw mask contains objects as category_ids
    raw_mask = self.load_annotation(self.anns[item])
    mask_instances, cats = self.get_instance_masks(raw_mask)

    tensors_resized = resize({"image": raw_frame, "mask": mask_instances.astype(np.uint8)},
                             self.resize_mode, self.crop_size)
    raw_frames, masks_instances = self.generate_clip(tensors_resized['image'], tensors_resized['mask'])
    raw_frames = np.transpose(raw_frames, axes=(3, 0, 1, 2))
    masks_instances = masks_instances[None]
    masks_fg = (masks_instances != 0).astype(np.uint8)
    # generate semantic segmentation mask using category ids
    masks_cat_ids = masks_instances.copy()
    for i, cat in enumerate(cats):
      masks_cat_ids[masks_cat_ids == i+1] = cats[i]

    return masks_cat_ids, masks_fg, masks_instances, raw_frames

  def set_video_id(self, video):
    pass

  def get_video_ids(self):
    return [0]

  def __len__(self):
    return len(self.anns)

  def __getitem__(self, item):
    info = {}
    info['name'] = self.name
    info['num_frames'] = self.__len__()

    masks_cat_ids, masks_fg, masks_instances, raw_frames = self.process_input(item)

    info['num_objects'] = len(np.unique(masks_instances))

    # padding size to be divide by 32
    _, _, h, w = masks_instances.shape
    new_h = h + 32 - h % 32 if h % 32 > 0 else h
    new_w = w + 32 - w % 32 if w % 32 > 0 else w
    # print(new_h, new_w)
    lh, uh = (new_h - h) / 2, (new_h - h) / 2 + (new_h - h) % 2
    lw, uw = (new_w - w) / 2, (new_w - w) / 2 + (new_w - w) % 2
    lh, uh, lw, uw = int(lh), int(uh), int(lw), int(uw)
    pad_masks = np.pad(masks_fg, ((0, 0), (0, 0), (lh, uh), (lw, uw)), mode='constant')
    pad_masks_sem_seg = np.pad(masks_cat_ids, ((0, 0), (0, 0), (lh, uh), (lw, uw)), mode='constant')
    pad_masks_instances = np.pad(masks_instances, ((0, 0), (0, 0), (lh, uh), (lw, uw)), mode='constant')
    pad_frames = np.pad(raw_frames, ((0, 0), (0, 0), (lh, uh), (lw, uw)), mode='constant')
    info['pad'] = ((lh, uh), (lw, uw))

    return {'images': pad_frames, 'info': info,
            'target': pad_masks, "proposals": pad_masks, "raw_proposals": pad_masks,
            'raw_masks': pad_masks,
            'target_extra': {'sem_seg': pad_masks_sem_seg,
                             'similarity_raw_mask': pad_masks_instances}}


if __name__ == '__main__':
    dataset = MapillaryVideoDataset(MAPILLARY_ROOT, is_train=False, resolution='quarter', crop_size=[256, 448])
    result = dataset.__getitem__(107)
    print("cats: {}\n instances: {}\n image range: {}\nfgmask: {}".format(np.unique(result['target_extra']['sem_seg']),
                                                                          np.unique(result['target_extra']['similarity_raw_mask']),
                                                                          (result['images'].min(), result['images'].max()),
                                                                          np.unique(result['raw_masks'])))
