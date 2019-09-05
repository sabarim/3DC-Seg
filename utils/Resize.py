import random
from enum import Enum, unique

import cv2
import numpy as np
from scipy.misc import imresize



@unique
class ResizeMode(Enum):
  FIXED_SIZE = "fixed_size"
  UNCHANGED = "unchanged"
  RANDOM_RESIZE_AND_CROP = "random_resize_and_crop"
  RANDOM_RESIZE_AND_OBJECT_CROP = "random_resize_and_object_crop"
  RESIZE_AND_OBJECT_CROP = "resize_and_object_crop"
  BBOX_CROP_AND_RESIZE_FIXED_SIZE = "bbox_crop_and_resize_fixed_size"
  RESIZE_MIN_SHORT_EDGE_MAX_LONG_EDGE = "resize_min_short_edge_max_long_edge"
  FIXED_RESIZE_AND_CROP = "fixed_resize_and_crop"
  AFFINE_WARP_TRAIN = "affine_warp_train"
  AFFINE_WARP_VALID = "affine_warp_valid"
  RESIZE_SHORT_EDGE = "resize_short_edge"
  RESIZE_SHORT_EDGE_AND_CROP = "resize_short_edge_and_crop"


def resize(tensors, resize_mode, size):
  if resize_mode == ResizeMode.UNCHANGED:
    return tensors

  crop_size = preprocess_size(size)
  if resize_mode == ResizeMode.RANDOM_RESIZE_AND_CROP:
    return random_resize_and_crop(tensors, crop_size)
  elif resize_mode == ResizeMode.RANDOM_RESIZE_AND_OBJECT_CROP:
    return random_resize_and_object_crop(tensors, crop_size)
  elif resize_mode == ResizeMode.RESIZE_AND_OBJECT_CROP:
    return resize_and_object_crop(tensors, crop_size)
  elif resize_mode == ResizeMode.FIXED_SIZE:
    return resize_fixed_size(tensors, crop_size)
  elif resize_mode == ResizeMode.RESIZE_SHORT_EDGE:
    return resize_short_edge_to_fixed_size(tensors, crop_size)
  elif resize_mode == ResizeMode.RESIZE_SHORT_EDGE_AND_CROP:
    return resize_short_edge_and_crop(tensors, crop_size)
  elif resize_mode == ResizeMode.BBOX_CROP_AND_RESIZE_FIXED_SIZE:
    return bbox_crop_and_resize_fixed_size(tensors, crop_size)
  else:
    assert False, ("resize mode not implemented yet", resize_mode)


def bilinear_resize(tensors, size):
  assert len(size) in (1, 2)
  if len(size) == 2:
    assert size[0] == size[1]
    crop_size = size
  else:
    crop_size = [size, size]


def random_resize_and_crop(tensors, size):
  tensors_resized = resize_random_scale_with_min_size(tensors, min_size=min(size))
  tensors_resized = random_crop_tensors(tensors_resized, size)
  return tensors_resized


def random_resize_and_object_crop(tensors, size):
  tensors_resized = resize_random_scale_with_min_size(tensors, min_size=min(size))
  tensors_resized = random_object_crop_tensors(tensors_resized, size)
  return tensors_resized

def resize_and_object_crop(tensors, size):
  tensors_resized = resize_random_scale_with_min_size(tensors, min_size=min(size))
  tensors_resized = random_object_crop_tensors(tensors_resized, size)
  return tensors_resized


def resize_short_edge_and_crop(tensors, size):
  tensors_resized = resize_short_edge_to_fixed_size(tensors, size)
  # TODO: the crop size is harcoded
  tensors_resized = random_crop_tensors(tensors_resized, (256, 455))
  return tensors_resized


def preprocess_size(size):
  size = tuple([size]) if isinstance(size, int) else tuple(size)
  assert len(size) in (1, 2)
  if len(size) == 2:
    # assert size[0] == size[1]
    crop_size = size
  else:
    crop_size = [size[0], size[0]]
  return crop_size


def resize_random_scale_with_min_size(tensors, min_size, min_scale=0.7, max_scale=1.3):
  assert min_size is not None
  img = tensors["image"]

  h = img.shape[0]
  w = img.shape[1]
  shorter_side = np.min([h, w])
  min_scale_factor = min_size / shorter_side
  min_scale = np.max([min_scale, min_scale_factor])
  max_scale = np.max([max_scale, min_scale_factor])
  scale_factor = random.uniform(min_scale, max_scale)
  scaled_size = np.around(np.array(img.shape[:2]) * scale_factor).astype(np.int)
  tensors_out = resize_fixed_size(tensors, scaled_size)
  return tensors_out


def scale_with_min_size(tensors, min_size, min_scale=0.7, max_scale=1.3):
  assert min_size is not None
  img = tensors["image"]

  h = img.shape[0]
  w = img.shape[1]
  shorter_side = np.min([h, w])
  min_scale_factor = float(min_size) / float(shorter_side)
  scaled_size = np.around(np.array(img.shape[:2]) * min_scale_factor).astype(np.int)
  tensors_out = resize_fixed_size(tensors, scaled_size)
  return tensors_out


def resize_short_edge_to_fixed_size(tensors, size):
  img = tensors["image"]

  h = img.shape[0]
  w = img.shape[1]
  shorter_side = np.min([h, w])
  scale_factor = size[0] / shorter_side
  scaled_size = np.around(np.array(img.shape[:2]) * scale_factor).astype(np.int)
  tensors_out = resize_fixed_size(tensors, scaled_size)
  return tensors_out

def random_crop_tensors(tensors, crop_size):
  assert "image" in tensors
  h, w = tensors["image"].shape[:2]
  new_h, new_w = crop_size

  top = int(random.uniform(0, max(h - new_h, 0)))
  left = int(random.uniform(0, max(w - new_w, 0)))

  tensors_cropped = {}
  for key in tensors.keys():
    tensors_cropped[key] = tensors[key][top: top + new_h,
                           left: left + new_w]

  return tensors_cropped


def random_object_crop_tensors(tensors, crop_size):
  assert "image" in tensors and 'mask' in tensors
  h, w = tensors["image"].shape[:2]
  new_h, new_w = crop_size
  obj_h_max = np.max(np.where(tensors['mask'] != 0)[0]) if tensors['mask'].sum() > 0 else 0
  obj_h_min = np.min(np.where(tensors['mask'] != 0)[0]) if tensors['mask'].sum() > 0 else 0
  obj_w_max = np.max(np.where(tensors['mask'] != 0)[1]) if tensors['mask'].sum() > 0 else 0
  obj_w_min = np.max(np.where(tensors['mask'] != 0)[1]) if tensors['mask'].sum() > 0 else 0

  top_lower_bound = max(0, obj_h_max - new_h)
  left_lower_bound = max(0, obj_w_max - new_w)
  top_upper_bound = min(max(0, (h - new_h)), obj_h_min)
  left_upper_bound = min(max(0, w - new_w), obj_w_min)

  top = int(random.uniform(top_lower_bound, top_upper_bound))
  left = int(random.uniform(left_lower_bound, left_upper_bound))

  tensors_cropped = {}
  for key in tensors.keys():
    tensors_cropped[key] = tensors[key][top: top + new_h,
                           left: left + new_w]

  return tensors_cropped



def resize_fixed_size(tensors, size):
  tensors_resized = {}
  for key in tensors.keys():
    tensor = tensors[key]
    if len(tensor.shape) > 2:
      tensors_resized[key] = imresize(tensor, size=size, interp='bilinear')
    else:
      tensors_resized[key] = imresize(tensor, size=size, interp='nearest')
      # if key in tensors_resized:
      #   # opencv accepts size in the form of (cols x rows), hence reverse the size
      #   tensors_resized[key] = cv2.resize(tensors_resized[key], tuple([size[1], size[0]]),
      #                                     interpolation = cv2.INTER_NEAREST)
  return tensors_resized


def bbox_crop_and_resize_fixed_size(tensors, size):
  MARGIN = 50
  tensors_cropped = tensors.copy()

  assert 'mask' in tensors
  locations = np.where(tensors['mask'] != 0)
  if len(locations[0]) > 0:
    shape = tensors['mask'].shape
    y0 = np.min(locations[0])
    y1 = np.max(locations[0])
    x0 = np.min(locations[1])
    x1 = np.max(locations[1])

    # add margin and clip to bounds
    y0 = np.maximum(y0 - MARGIN, 0)
    x0 = np.maximum(x0 - MARGIN, 0)
    y1 = np.minimum(y1 + MARGIN, shape[0])
    x1 = np.minimum(x1 + MARGIN, shape[1])

    for key in tensors:
      tensors_cropped[key] = tensors[key][y0:y1, x0:x1]

  tensors_cropped = resize_fixed_size(tensors_cropped, size)
  return tensors_cropped