import torch

from util import get_best_overlap


THRESH=0.15


def stitch_with_gt(curr_tube, target_one_hot):
  """
  :param curr_predictions: dictionary of current predictions [{'mask': prediction}, ...]
  :param target_one_hot: one hot target annotations (T*H*W) excluding the background
  """
  gt_aligned_tube = torch.zeros_like(curr_tube).int()
  objects = torch.nonzero(curr_tube.unique())
  unused_objs = []
  for obj in objects:
    best_iou, target_id = get_best_overlap((curr_tube == obj).int().data.cpu().numpy(), target_one_hot.data.cpu().numpy())
    if best_iou >  THRESH:
      gt_aligned_tube[curr_tube == obj] = torch.tensor(target_id + 1).int()
    else:
      unused_objs += [obj]

  for obj in unused_objs:
    gt_aligned_tube[curr_tube == obj] = gt_aligned_tube.max() + 1

  return gt_aligned_tube



