import glob
import pickle
import torch
import numpy as np
import os
import multiprocessing as mp
from lib.flowlib import read_flow
from scripts.eval_maskrcnn_warp import get_iou
from scripts.warp_davis_maskrcnn import warp

from scripts.path_constants import DAVIS_ROOT
from util import top_n_predictions_maskrcnn


MAX_PROPOSALS = 20

maskrcnn_data_dir = "../results/converted_proposals/thresh-0/"
warped_data_dir = "../results/maskrcnn_warped"
# davis_data_dir = '/globalwork/data/DAVIS-Unsupervised/DAVIS/'
# davis_data_dir = '/globalwork/data/DAVIS-Unsupervised//DAVIS-test/DAVIS/'
davis_data_dir = '/globalwork/data/DAVIS-Unsupervised/DAVIS-2019-Unsupervised-test-dev-480p//DAVIS/'
flow_dir = "/globalwork/data/DAVIS-Unsupervised/DAVIS-2019-Unsupervised-test-dev-480p/DAVIS/flo/"
out_dir = "/globalwork/data/DAVIS-Unsupervised//DAVIS-2019-Unsupervised-test-dev-480p/DAVIS/warp_data/"


def warp_all(associated_proposals, flo):
  masks = associated_proposals['masks']
  if len(masks) > 0:
    warped_masks = torch.cat([warp(mask.unsqueeze(0).float().cuda(),
                                   flo.unsqueeze(0).permute(0, -1, 1, 2).cuda())
                              for mask in masks], dim=0)
  return warped_masks


def get_iou_scores(ref_objs, proposals):
  result_scores = torch.zeros(len(ref_objs), len(proposals['masks']))
  for ref_id in range(len(ref_objs)):
    iou_scores = [get_iou(ref_objs[ref_id, 0].int().data.cpu().numpy(),
                          proposals['masks'][obj_id][0].data.cpu().numpy().astype(np.uint8))
                  for obj_id in range(len(proposals['masks']))]
    result_scores[ref_id] = torch.tensor(iou_scores)

  return result_scores


def read_proposals(i, seq):
  p1_path = os.path.join(maskrcnn_data_dir, seq, '{:05d}'.format(i) + ".pickle")
  p2_path = os.path.join(maskrcnn_data_dir, seq, '{:05d}'.format(i + 1) + ".pickle")

  p1 = pickle.load(open(p1_path, 'rb'))
  p1 = top_n_predictions_maskrcnn(p1, MAX_PROPOSALS)

  p2 = pickle.load(open(p2_path, 'rb'))
  p2 = top_n_predictions_maskrcnn(p2, MAX_PROPOSALS)

  return p1, p2


def save_scores(seq):
  seq = seq.rstrip()
  out_folder = out_dir + seq
  if not os.path.exists(out_folder):
    os.makedirs(out_folder)
  all_proposals = glob.glob(os.path.join(maskrcnn_data_dir, seq, "*.pickle"))

  for i in range(len(all_proposals)-1):
    flo = torch.from_numpy(read_flow(os.path.join(flow_dir, seq, '{:05d}'.format(i + 1) + ".flo"))).float()
    p1, p2 = read_proposals(i, seq)
    warped_proposals = warp_all(p1, flo)

    p1_associations = {}
    p1_associations['warped_masks'] = warped_proposals
    p1_associations['iou_scores'] = get_iou_scores(warped_proposals, p2)
    p1_out = os.path.join(out_folder, '{:05d}'.format(i) + ".pickle")
    print("pickling {}".format(p1_out))
    pickle.dump(p1_associations, open(p1_out, 'wb'))


def main():
  seqs = DAVIS_ROOT + "ImageSets/2019/test-dev.txt"
  lines = ['juggling-selfie']
  # pool = mp.Pool(2)
  # with open(os.path.join(seqs), "r") as lines:
  #  pool.map(save_scores, [line for line in lines])
  for line in lines:
   save_scores(line)


if __name__ == '__main__':
  main()