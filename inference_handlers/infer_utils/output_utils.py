import numpy as np


def get_class_label(mask, labels):
  """

  :param mask: bool array representing the object instance
  :param labels: semantic segmentation mask
  :return: 
  """
  best_overlap = 0
  target_label = 0

  overlap = labels[mask]

  for _id in np.unique(overlap):
    if _id == 0:
      continue
    numel = (overlap == _id).sum()
    if numel > best_overlap:
      best_overlap = numel
      target_label = _id

  return target_label