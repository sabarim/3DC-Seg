import importlib
import pkgutil
import numpy as np
from imgaug import augmenters as iaa

import sys

TRANSLATION = 0.1
SHEAR = 0.05
ROTATION = 15
FLIP = 40


def import_submodules(package_name):
  package = sys.modules[package_name]
  for importer, name, is_package in pkgutil.walk_packages(package.__path__):
    # not sure why this check is necessary...
    if not importer.path.startswith(package.__path__[0]):
      continue
    name_with_package = package_name + "." + name
    importlib.import_module(name_with_package)
    if is_package:
      import_submodules(name_with_package)


def generate_clip_from_image(raw_frame, raw_mask, temporal_window, **kwargs):
  """

  :param raw_frame: The frame to be augmented: h x w x 3
  :param raw_mask: h x w x 1
  :param temporal_window: Number of frames in the output clip
  :return: clip_frames - list of frames with values 0-255
           clip_masks - corresponding masks
  """
  global TRANSLATION, ROTATION, SHEAR
  if 'translation' in kwargs:
    TRANSLATION = kwargs['translation']
  if 'rotation' in kwargs:
    ROTATION = kwargs['rotation']
  if 'shear' in kwargs:
    SHEAR = kwargs['shear']

  clip_frames = np.repeat(raw_frame[np.newaxis], temporal_window, axis=0)
  clip_masks = np.repeat(raw_mask[np.newaxis], temporal_window, axis=0)
  # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
  # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second
  # image.
  sometimes = lambda aug: iaa.Sometimes(0.05, aug)
  blur = sometimes(iaa.OneOf([
    iaa.GaussianBlur((0.0, 0.5)),
    # iaa.AverageBlur(k=(2, 7)),
    # iaa.MedianBlur(k=(3, 11)),
  ]))
  seq = iaa.Sequential([
    # iaa.Fliplr(FLIP / 100.),  # horizontal flips
    sometimes(iaa.ElasticTransformation(alpha=(200, 220), sigma=(17.0, 19.0))),
    iaa.Affine(
      scale={"x": (0.7, 1.3), "y": (0.7, 1.3)},
      translate_percent={"x": (-TRANSLATION, TRANSLATION), "y": (-TRANSLATION, TRANSLATION)},
      rotate=(-ROTATION, ROTATION),
      shear=(-SHEAR, SHEAR),
      mode='edge',
    )
  ], random_order=True)

  frame_aug = raw_frame[np.newaxis]
  mask_aug = raw_mask[np.newaxis]
  # create sequence of transformations of the current image
  for t in range(temporal_window - 1):
    frame_aug, mask_aug = seq(images=frame_aug.astype(np.uint8), segmentation_maps=mask_aug.astype(np.uint8))
    frame_aug = blur(images=frame_aug)
    clip_frames[t + 1] = frame_aug[0]
    clip_masks[t + 1] = mask_aug[0]

  return clip_frames, clip_masks