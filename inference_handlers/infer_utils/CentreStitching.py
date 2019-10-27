import torch
from scipy.optimize import linear_sum_assignment


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
  ref_centres = torch.cat([p['centre'] for p in ref_predictions], dim=1)
  result = {}
  # for curr_pred in [p for p in curr_predictions]:
  dist = torch.stack([torch.sum(torch.pow(ref_centres - curr_pred['centre'], 2), dim=0) for curr_pred in curr_predictions])
  row_ind, col_ind = linear_sum_assignment(dist.data.cpu())
  for (curr_idx, ref_idx) in zip(row_ind, col_ind):
    result[ref_idx + 1] = curr_predictions[curr_idx]
    # matched_ref = torch.argmin(dist)
    # result[matched_ref+1] = curr_pred

  for idx in range(len(curr_predictions)):
    if idx not in col_ind and idx + 1 not in result.keys():
      result[idx+1] = curr_predictions[idx]
    elif idx not in col_ind:
      result[max(result.keys()) + 1] = curr_predictions[idx]

  return result