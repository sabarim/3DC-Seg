import os
import glob
import pickle
import numpy as np
import multiprocessing as mp

import torch
from PIL import Image

from lib.flowlib import read_flow
from scripts.warp_davis_maskrcnn import warp
from util import save_mask, write_output_mask, get_one_hot_vectors, top_n_predictions_maskrcnn

CONF_THRESH=0.0
IOU_THRESH = 0.1
MAX_PROPOSALS=20

# flags to switch the reid modules
USE_ORACLE_REID = False
USE_REID = False

# maskrcnn_data_dir = "/globalwork/mahadevan/mywork/data/training/pytorch/forwarded/maskrcnn/thresh-0.7/"
maskrcnn_data_dir = "/globalwork/mahadevan/data/davis_unsupervised_results/converted_proposals/thresh-0-all_fields/"
warped_data_dir = "/globalwork/mahadevan/data/davis_unsupervised_results/results/maskrcnn_warped"
davis_data_dir = '/globalwork/data/DAVIS-Unsupervised/DAVIS/'
#davis_data_dir = '/globalwork/data/DAVIS-Unsupervised//DAVIS-test/DAVIS/'
# davis_data_dir = '/globalwork/data/DAVIS-Unsupervised/DAVIS-2019-Unsupervised-test-dev-480p//DAVIS/'
flow_dir = "/globalwork/data/DAVIS-Unsupervised/DAVIS/flo/"
out_dir = "results/eval_maskrcnn_warp/davis-val/"


def get_iou(gt, pred):
  i = np.logical_and(pred > 0, gt > 0).sum()
  u = np.logical_or(pred > 0, gt > 0).sum()
  if u == 0:
    iou = 1.0
  else:
    iou = i / u
  return iou


def warp_all(associated_proposals, flo, f, seq):
  masks = associated_proposals['mask']
  # pool = mp.Pool(4)
  # warped_masks = torch.cat([pool.apply(warp, args = (mask.unsqueeze(0).float().cuda(),
  #                                                    flo.unsqueeze(0).permute(0, -1, 1, 2).cuda()))
  #                           for mask in masks], dim=0)

  warped_masks_file = os.path.join(davis_data_dir, "warp_data", seq, '{:05d}'.format(f) + ".pickle")
  if os.path.exists(warped_masks_file):
    warped_masks = pickle.load(open(warped_masks_file, 'rb'))['warped_masks']
  elif len(masks)>0:
    warped_masks = torch.cat([warp(mask.unsqueeze(0).float().cuda(),
                                   flo.unsqueeze(0).permute(0, -1, 1, 2).cuda())
                              for mask in masks], dim=0)
  else:
    warped_masks = masks
  associated_proposals['mask'] = warped_masks
  return associated_proposals


def get_best_match(ref_obj, proposals):
  best_iou = 0
  target_id = -1
  mask = None
  for obj_id in range(len(proposals)):
    iou = get_iou(ref_obj[0].int().data.cpu().numpy(), proposals[obj_id][0].data.cpu().numpy().astype(np.uint8))
    if iou > best_iou and iou > IOU_THRESH:
      best_iou = iou
      target_id = obj_id
      mask = proposals[obj_id]

  return mask, best_iou, target_id


def get_initial_proposal(line, out_folder):
  initial_proposals = os.path.join(maskrcnn_data_dir, line, '{:05d}'.format(0) + ".pickle")
  proposals = pickle.load(open(initial_proposals, 'rb'))

  # TODO: select topn n instead of using conf thresh
  associated_proposals = top_n_predictions_maskrcnn(proposals, MAX_PROPOSALS)
  associated_proposals['track_ids'] = list(range(len(associated_proposals['mask'])))
  write_output_mask(associated_proposals, out_folder + '/{:05d}.png'.format(0))
  pickle.dump(associated_proposals, open(out_folder + '/{:05d}.pickle'.format(0), 'wb'))

  if USE_ORACLE_REID:
    # use gt for oracle reid
    gt_path = os.path.join(davis_data_dir, "Annotations_unsupervised/480p", line, '{:05d}.png'.format(0))
    gt = get_one_hot_vectors(np.array(Image.open(gt_path), dtype=np.uint16))
    gt_tracks = np.ones(len(gt))*-1

    # use ground truth tracks for oracle re-id.
    # FIXME: assume that all gt objects are available in the first frame for now
    for i in range(len(gt)):
      gt_mask, iou, id = get_best_match(torch.from_numpy(gt[i]).unsqueeze(0),
                                        associated_proposals['mask'])
      if gt_mask is None:
        print("WARN: GT object not found in the first frame proposals")
        gt_tracks[i] = np.max(associated_proposals["track_ids"]) + 1
      else:
        gt_tracks[i] = id

    # add gt track ids for oracle re-id
    associated_proposals["gt_tracks_ids"] = gt_tracks

  return associated_proposals


def save_tracklets(proposals, warped_proposals, out_folder, f):
  """
  
  :param proposals: BoxList 
  :param warped_proposals: dict - contains 'n' top predictions orgnanised as dict
                            dict[i] = {'masks':<nd array with binary mask>, 
                                       'score':<score of the prediction before warp>}
  :param out_folder: 
  :param f: 
  :return: 
  """
  # top_predictions = select_top_predictions(proposals, CONF_THRESH)
  print("proposal length", len(list(proposals.values())))
  shape = list(proposals['mask'].shape[2:])
  track_ids = np.ones_like(proposals['scores']) * -1
  ious = np.ones_like(proposals['scores']) * -1
  output_mask = np.zeros(shape)


  # get track associations based on re-id
  reid_tracks = get_reid_tracks(proposals, track_ids)

  for i in range(len(proposals['mask'])):
    proposal_mask = proposals['mask'][i]
    warped_mask, iou, id = get_best_match(proposal_mask, warped_proposals['mask'])
    reid_flag = reid_tracks[i] != -1

    if warped_mask is not None:
      # associate tracks based on warped proposals if there is no re-id score
      if not reid_flag:
        track_ids[i] = id if not 'track_ids' in warped_proposals else warped_proposals['track_ids'][id]
      else:
        # TODO: integrate re-id tracks based on the re-id score
        track_ids[i] = int(proposals['gt_tracks_ids'][int(reid_tracks[i])])
      output_mask[proposal_mask[0].data.cpu().numpy() == 1] = track_ids[i]+1
      ious[i] = iou
      # ids_chosen+=[id]

  max_track_id = np.max(track_ids) if len(track_ids) > 0 else 0
  if (track_ids == -1).sum() > 0:
    ids_not_associated = np.where(track_ids == -1)
    track_ids[track_ids==-1]=list(range(int(max_track_id)+1, int(max_track_id)+1 + (track_ids==-1).sum()))
    for index in ids_not_associated[0]:
      output_mask[proposals['mask'][index][0].data.cpu().numpy() == 1] = track_ids[index] + 1
  proposals['track_ids'] =  track_ids[track_ids < MAX_PROPOSALS]
  output_mask[output_mask>=20] = 0

  proposals['ious'] = ious
  out_file = os.path.join(out_folder, '{:05d}'.format(f + 1) + ".pickle")
  print("pickling {}".format(out_file))
  pickle.dump(proposals, open(out_file, 'wb'))

  out_file = os.path.join(out_folder, '{:05d}'.format(f + 1) + ".png")
  if np.max(output_mask) > 255:
    print("max value is greater than 255 for {}".format(out_file))
    output_mask[output_mask>255] = 0
  else:
    save_mask(output_mask.astype(np.int), out_file)

  return proposals


def get_reid_tracks(proposals, track_ids):
  gt_ids = np.ones_like(proposals['scores']) * -1
  if USE_ORACLE_REID:
    gt_ids = oracle_reid(proposals, track_ids)
  elif USE_REID:
    gt_ids = reid(proposals, track_ids)
  return gt_ids


def oracle_reid(proposals, track_ids):
  gt_ids = np.ones_like(proposals['scores']) * -1
  # use ground truth tracks for oracle re-id
  for i in range(len(proposals['gt_masks'])):
    gt_mask, iou, id = get_best_match(torch.from_numpy(proposals['gt_masks'][i]).unsqueeze(0),
                                      proposals['masks'])
    if gt_mask is not None:
      gt_ids[id] = i
      track_ids[id] = i
  return gt_ids

def reid():
  pass


def run_eval(line):
  # if line.rstrip() not in ["trucks-race","mantaray","mascot","motorbike-race", "obstacles"]:
  #   return
  print(line)
  line = line.rstrip()
  out_folder = out_dir + line
  if not os.path.exists(out_folder):
    os.makedirs(out_folder)
  all_proposals = glob.glob(os.path.join(maskrcnn_data_dir, line, "*.pickle"))

  # generate the proposals for the first frame
  associated_proposals = get_initial_proposal(line, out_folder)

  for i in range(len(all_proposals) - 1):
    # warped_proposals_path = os.path.join(warped_data_dir, line, '{:05d}'.format(i + 1) + ".pickle")
    # warped_proposals = pickle.load(open(warped_proposals_path, 'rb'))
    # warp the proposals
    flo = torch.from_numpy(read_flow(os.path.join(flow_dir, line, '{:05d}'.format(i+1) + ".flo"))).float()
    warped_proposals = warp_all(associated_proposals, flo, i, line)

    proposals_raw_path = os.path.join(maskrcnn_data_dir, line, '{:05d}'.format(i+1) + ".pickle")
    proposals_raw = pickle.load(open(proposals_raw_path, 'rb'))
    proposals_raw = top_n_predictions_maskrcnn(proposals_raw, MAX_PROPOSALS)

    if USE_ORACLE_REID:
      # use gt for oracle reid
      gt_path = os.path.join(davis_data_dir, "Annotations_unsupervised/480p", line, '{:05d}.png'.format(i + 1))
      gt = get_one_hot_vectors(np.array(Image.open(gt_path), dtype=np.uint16))
      proposals_raw['gt_masks'] = gt
      proposals_raw['gt_tracks_ids'] = warped_proposals['gt_tracks_ids']

    associated_proposals = save_tracklets(proposals_raw, warped_proposals, out_folder, i)


def main():
  seqs = davis_data_dir + "ImageSets/2017/val.txt"
  lines = ['bmx-trees']
  pool = mp.Pool(1)
  # with open(os.path.join(seqs), "r") as lines:
  #  pool.map(run_eval, [line for line in lines])
  for line in lines:
    run_eval(line)


if __name__ == '__main__':
  main()
