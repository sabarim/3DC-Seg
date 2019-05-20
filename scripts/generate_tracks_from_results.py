import glob
import os

import pickle
import numpy as np
import multiprocessing as mp

import torch
from PIL import Image

from scripts.eval_maskrcnn_warp import get_best_match
from scripts.path_constants import OUT_DIR, DAVIS_ROOT
from util import save_mask, get_one_hot_vectors


def save_images_for_eval(proposals, out_folder, f):
  shape = proposals.get_field("mask").shape[2:]
  output_mask = np.zeros(shape)
  gt_ids = []
  track_ids = proposals.get_field('track_ids')

  for i in range(len(proposals.get_field('gt_masks'))):
    gt_mask, iou, id = get_best_match(torch.from_numpy(proposals.get_field('gt_masks')[i]).unsqueeze(0),
                                      proposals.get_field('mask'))
    if gt_mask is not None:
      output_mask[proposals.get_field('mask')[id,0].data.cpu().numpy() == 1] = i + 1

  out_file = os.path.join(out_folder, '{:05d}'.format(f) + ".png")
  save_mask(output_mask.astype(np.int), out_file)


def run_eval(line):
  print(line)
  line = line.rstrip()
  in_folder = OUT_DIR + line
  out_folder = OUT_DIR + "/gt_tracks/" + line
  if not os.path.exists(out_folder):
    os.makedirs(out_folder)
  all_proposals = glob.glob(os.path.join(in_folder, "*.pickle"))

  for i in range(len(all_proposals)):
    proposals_raw = pickle.load(open(all_proposals[i], 'rb'))
    # use gt for oracle reid
    gt_path = os.path.join(DAVIS_ROOT, "Annotations_unsupervised/480p", line, '{:05d}.png'.format(i))
    gt = get_one_hot_vectors(np.array(Image.open(gt_path), dtype=np.uint16))
    proposals_raw.add_field('gt_masks', gt)
    save_images_for_eval(proposals_raw, out_folder, i)


def main():
  seqs = DAVIS_ROOT + "ImageSets/2017/val.txt"
  lines = ['bmx-trees']
  pool = mp.Pool(5)
  with open(os.path.join(seqs), "r") as lines:
   pool.map(run_eval, [line for line in lines])
  #for line in lines:
  #  run_eval(line)


if __name__ == '__main__':
  main()
