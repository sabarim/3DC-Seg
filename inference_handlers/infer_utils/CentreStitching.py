import torch
import numpy as np
from scipy.optimize import linear_sum_assignment


THRESHOLD = 0.007


def stitch_centres(ref_predictions, curr_predictions, tube_shape):
  result = associate_centres(curr_predictions, ref_predictions)
  stitched_instances = torch.zeros(tube_shape)
  for key, value  in result.items():
    stitched_instances[value['mask'] != 0] = float(key)

  result_pred = [result[key] for key in sorted(result.keys())]
  return stitched_instances, result_pred


"""
:returns dict {ref_idx: curr_pred}
"""
def associate_centres(curr_predictions, ref_predictions):
  result = {}
  if len(curr_predictions) == 0:
    return result
  elif len(ref_predictions) == 0:
    ref_predictions = curr_predictions
  ref_centres = torch.cat([p['centre'] for p in ref_predictions], dim=1)
  # for curr_pred in [p for p in curr_predictions]:
  dist = torch.stack([torch.sum(torch.pow(ref_centres - curr_pred['centre'], 2), dim=0) for curr_pred in curr_predictions])
  row_ind, col_ind = linear_sum_assignment(dist.data.cpu())
  # print("Average distance: {}", dist[row_ind, col_ind].mean())
  # track predictions that are associated
  ids_used = []
  for (curr_idx, ref_idx) in zip(row_ind, col_ind):
  # for (curr_idx, ref_idx) in zip(torch.where(dist < THRESHOLD)[0],torch.where(dist < THRESHOLD)[1]):
    if dist[curr_idx, ref_idx] < THRESHOLD:
      result[ref_idx + 1] = curr_predictions[curr_idx]
      ids_used += [curr_idx]

  # ids that were not associated by linear assignment
  ids_not_associated = np.setdiff1d(np.arange(len(curr_predictions)), np.array(ids_used))
  for a in ids_not_associated:
    # if distance is less than threshold, then merge instances
    if torch.min(dist[a]) < THRESHOLD*2:
      ref_idx = torch.argmin(dist[a]).item() + 1
      if ref_idx in result.keys():
        result[ref_idx]['mask']  += curr_predictions[a]['mask']
        result[ref_idx]['mask'] = (result[ref_idx]['mask'] != 0).int()
      else:
        result[ref_idx] = curr_predictions[a]
    else:
      curr_idx = a+1
      if curr_idx not in result.keys():
        result[curr_idx] = curr_predictions[a]
      else:
        result[max(result.keys()) + 1] = curr_predictions[a]

  return result
