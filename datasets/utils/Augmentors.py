import numpy as np
from datasets.utils.OclussionAug import load_occluders, occlude_with_objects

occ_aug = None


def load_augmentors(args, pascal_voc_path):
  if args is None:
    return
  augmentors = []
  if 'occ' in args:
    augmentors += {'occluders': load_occluders(pascal_voc_path)}

  return augmentors


def augment(augmentors, aug_classes, tensors):
  tensors = tensors.copy()

  if 'occ' in augmentors:
    assert 'occluders' in aug_classes
    tensors = do_occ_aug(aug_classes['occluders'], tensors)

  return tensors


def do_occ_aug(occluders, tensors, p=0.2):
  occluded_tensors = tensors.copy()
  if np.random.choice([True, False], 1, p=[p, 1-p]):
    occluded_tensors = occlude_with_objects(occluded_tensors, occluders)

  return occluded_tensors