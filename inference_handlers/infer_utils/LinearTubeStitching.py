import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from torch import __init__


def get_best_overlap(ref_tube, curr_tube):
  THRESH = 0.7
  num_obj_curr = curr_tube.max()
  num_obj_ref = ref_tube.max()
  ref_tube = ref_tube.data.cpu().numpy()
  curr_tube = curr_tube.data.cpu().numpy()
  conf_matrix = confusion_matrix(curr_tube.reshape(-1), ref_tube.reshape(-1))[1:, 1:]
  I = conf_matrix
  U = conf_matrix.sum(axis=0)  + conf_matrix.sum(axis=1)[:, None] - np.diag(conf_matrix)
  iou = np.nan_to_num(I / U)
  cost = 1-iou
  cost = cost[:, :num_obj_ref]
  # filter background
  # compute linear assignment for foreground objects
  row_ind, col_ind = linear_sum_assignment(cost)

  ids_to_delete = np.where(cost[row_ind, col_ind] > THRESH)
  row_ind = np.delete(row_ind, ids_to_delete)
  col_ind = np.delete(col_ind, ids_to_delete)
  unassigned_objects = np.setdiff1d(np.arange(curr_tube.max() - 1), np.array(row_ind))

  for obj_id in unassigned_objects:
    row_ind= np.append(row_ind, [obj_id])
    if len(cost) > 0 and np.min(cost[obj_id]) < THRESH:
      ref_id = np.argmin(cost[obj_id])
    elif obj_id not in col_ind:
      ref_id = obj_id
    else:
      ref_id = np.max(col_ind) + 1
    col_ind = np.append(col_ind, [ref_id])
  # col_ind[cost[row_ind, col_ind] > THRESH] = -1
  return row_ind + 1, col_ind + 1


def get_best_overlap_iou(ref_tube, curr_tube):
  THRESH = 0.7
  num_obj_curr = curr_tube.max()
  num_obj_ref = ref_tube.max()
  ref_tube = ref_tube.data.cpu().numpy()
  curr_tube = curr_tube.data.cpu().numpy()
  conf_matrix = confusion_matrix(curr_tube.reshape(-1), ref_tube.reshape(-1))[1:, 1:]
  I = conf_matrix
  U = conf_matrix.sum(axis=0)  + conf_matrix.sum(axis=1) - np.diag(conf_matrix)
  iou = np.nan_to_num(I / U[None])
  cost = 1-iou
  cost = cost[:, :num_obj_ref]
  iou = iou[:num_obj_curr, :num_obj_ref]
  col_ind = np.argmax(iou, axis=1)
  row_ind = np.arange(len(col_ind))
  selected_ious = np.max(iou, axis=1)

  # Assign new ids for objects with low overlap
  n_objects_unassigned = np.sum(selected_ious < THRESH)
  available_ids = np.setdiff1d(row_ind, col_ind)
  new_ids_count = n_objects_unassigned - len(available_ids)
  if(new_ids_count > 0):
    available_ids = np.append(available_ids, np.arange(num_obj_ref, num_obj_ref + new_ids_count))
  available_ids.sort()
  col_ind[selected_ious < THRESH] = available_ids[:n_objects_unassigned]
  return row_ind + 1, col_ind + 1


def get_overlapping_proposals(ref_tube, curr_tube, overlaps):
  shape = curr_tube.shape
  ref_tube_overlap = ref_tube[8-overlaps:]
  curr_tube_overlap = curr_tube[:overlaps]

  # target ids contain indices of the chosen track ids
  curr_ids, target_ids = get_best_overlap(ref_tube_overlap, curr_tube_overlap)

  # store the current and reference track ids
  obj_ids_ref = np.arange(ref_tube_overlap.unique().int().max() + 1)
  obj_ids_ref.sort()
  # obj_ids_ref = obj_ids_ref[obj_ids_ref!=0]
  obj_ids_curr = np.arange(curr_tube_overlap.unique().int().max() + 1)
  obj_ids_curr.sort()
  stitched_tube = torch.zeros_like(curr_tube).int()

  for curr_idx, ref_idx in zip(curr_ids, target_ids):
    stitched_tube[curr_tube == curr_idx] = torch.tensor(ref_idx, dtype=torch.int)
    # if ref_idx >= len(obj_ids_ref) and ref_idx < len(obj_ids_curr):
    #   stitched_tube[curr_tube.int() == obj_ids_curr[curr_idx]] = torch.tensor(obj_ids_curr[ref_idx]).int()
    # if ref_idx >= len(obj_ids_ref):
    #   stitched_tube[curr_tube.int() == obj_ids_curr[curr_idx]] = torch.tensor(ref_idx).int()
    # elif curr_idx<len(obj_ids_curr):
    # else:
    #   stitched_tube[curr_tube.int() == obj_ids_curr[curr_idx]] = torch.tensor(obj_ids_ref[ref_idx]).int()


  # for idx in range(len(obj_ids_ref)):
  #   # reference track id to be used for replacement
  #   track_id = obj_ids_ref[idx]
  #   if track_id == 0: #background
  #     continue
  #   target_idx = target_ids[idx]
  #   id_to_replace = obj_ids_curr[target_idx] if target_idx < len(obj_ids_curr) else -1
  #
  #   if id_to_replace !=0 and id_to_replace != -1:
  #     stitched_tube[curr_tube.int() == id_to_replace] = track_id
  #     obj_ids_curr[target_idx] = -1
  #
  # for obj_id in obj_ids_curr:
  #   if obj_id not in [0, -1]:
  #     stitched_tube[curr_tube.int() == obj_id] = stitched_tube.max()+1
  return stitched_tube


def stitch_clips_best_overlap(last_predictions, curr_predictions, overlaps):
  stitched_tube = get_overlapping_proposals(last_predictions, curr_predictions, overlaps)
  return stitched_tube