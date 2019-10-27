import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from torch import __init__



def get_best_overlap(ref_tube, curr_tube):
  THRESH = 0.9
  ref_tube = ref_tube.data.cpu().numpy()
  curr_tube = curr_tube.data.cpu().numpy()
  conf_matrix = confusion_matrix(ref_tube.flatten(), curr_tube.flatten())
  cost = 1-np.nan_to_num(conf_matrix /
                         (conf_matrix.sum(axis=1) + conf_matrix.sum(axis=0) - np.diag(conf_matrix))[None, :])
  # filter background
  # cost = cost[1:, 1:]
  # compute linear assignment for foreground objects
  row_ind, col_ind = linear_sum_assignment(cost)
  col_ind[cost[row_ind, col_ind] > THRESH] = -1
  return col_ind + 1


def get_overlapping_proposals(ref_tube, curr_tube, overlaps):
  shape = curr_tube.shape
  ref_tube_overlap = ref_tube[8-overlaps:].reshape((overlaps, -1)).contiguous()
  curr_tube_overlap = curr_tube[:overlaps].reshape((overlaps,-1)).contiguous()

  # target ids contain indices of the chosen track ids
  target_ids = get_best_overlap(ref_tube_overlap, curr_tube_overlap)

  # store the current and reference track ids
  obj_ids_ref = ref_tube_overlap.unique().int()
  obj_ids_ref.sort()
  # obj_ids_ref = obj_ids_ref[obj_ids_ref!=0]
  obj_ids_curr = curr_tube_overlap.unique().int()
  obj_ids_curr.sort()
  stitched_tube = torch.zeros_like(curr_tube)

  for idx in range(len(obj_ids_ref)):
    # reference track id to be used for replacement
    track_id = obj_ids_ref[idx]
    if track_id == 0: #background
      continue
    target_idx = target_ids[idx]
    id_to_replace = obj_ids_curr[target_idx] if target_idx < len(obj_ids_curr) else -1

    if id_to_replace !=0 and id_to_replace != -1:
      stitched_tube[curr_tube.int() == id_to_replace] = track_id
      obj_ids_curr[target_idx] = -1

  for obj_id in obj_ids_curr:
    if obj_id not in [0, -1]:
      stitched_tube[curr_tube.int() == obj_id] = stitched_tube.max()+1
  return stitched_tube


def stitch_clips_best_overlap(last_predictions, curr_predictions, overlaps):
  stitched_tube = get_overlapping_proposals(last_predictions, curr_predictions, overlaps)
  return stitched_tube