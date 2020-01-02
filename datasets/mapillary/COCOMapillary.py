from torch.utils.data.dataset import Dataset

from datasets.coco.COCO import COCOEmbeddingDataset
from datasets.mapillary.MapillaryInstance import MapillaryVideoDataset
from utils.Constants import COCO_ROOT, MAPILLARY_ROOT
from utils.Resize import ResizeMode


class COCOMapillary(Dataset):
  def __init__(self, is_train=False, crop_size=None,temporal_window=8, resize_mode=ResizeMode.FIXED_SIZE, min_size=0):
    self.coco_dataset = COCOEmbeddingDataset(COCO_ROOT, is_train=is_train, crop_size=crop_size,temporal_window=temporal_window,
                                        resize_mode=resize_mode)
    self.mapillary_dataset = MapillaryVideoDataset(MAPILLARY_ROOT, is_train=is_train, resolution='quarter',
                                              crop_size=crop_size, temporal_window=temporal_window,
                                              resize_mode=resize_mode, min_size=min_size)

  def __len__(self):
    return self.coco_dataset.__len__() + self.mapillary_dataset.__len__()

  def __getitem__(self, item):
    if item < self.coco_dataset.__len__():
      return self.coco_dataset.__getitem__(item)
    else:
      mapillary_item = item - self.coco_dataset.__len__()
      return self.mapillary_dataset.__getitem__(mapillary_item)
