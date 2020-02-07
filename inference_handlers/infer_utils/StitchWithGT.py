import torch

from utils.util import get_iou

THRESH=0.1


def stitch_with_gt(curr_tube, target, num_objects):
  """
  :param curr_predictions: dictionary of current predictions [{'mask': prediction}, ...]
  :param target_one_hot: one hot target annotations (T*H*W) excluding the background
  :param num_objects: number of objects in the sequence
  """
  gt_aligned_tube = torch.zeros_like(curr_tube).int()
  objects = torch.nonzero(curr_tube.unique())
  unused_objs = []
  for obj in objects:
    best_iou, target_id = get_best_overlap((curr_tube == obj).int(), target)
    if best_iou >  THRESH and target_id != -1:
      gt_aligned_tube[curr_tube == obj] = torch.tensor(target_id).int()
    else:
      unused_objs += [obj]

  for obj in unused_objs:
    gt_aligned_tube[curr_tube == obj] = max(gt_aligned_tube.max(), num_objects) + 1

  return gt_aligned_tube


def get_best_overlap(ref_obj, target):
    best_iou = 0
    target_id = -1
    # mask = proposals[:, 0].cuda()

    for obj_id in target.unique():
      iou = get_iou(ref_obj.byte().data.cpu().numpy(), (target == obj_id).byte().data.cpu().numpy())
      if iou > best_iou:
        best_iou = iou
        target_id = obj_id
        # mask = (proposals[:, 0] == obj_id).int().cuda()

    return best_iou, target_id
