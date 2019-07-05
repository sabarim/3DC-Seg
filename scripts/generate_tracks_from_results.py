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


USE_ORACLE_REID=False
root = '/globalwork/data/DAVIS-Unsupervised/DAVIS-2019-Unsupervised-test-dev-480p/DAVIS/'

def save_images_for_eval(proposals, out_folder, f, output_tracks, last_update):
  shape = proposals["masks"].shape[2:]
  output_mask = np.zeros(shape)
  gt_ids = []
  track_ids = proposals['track_ids']

  if USE_ORACLE_REID:
    for i in range(len(proposals.get_field('gt_masks'))):
      gt_mask, iou, id = get_best_match(torch.from_numpy(proposals.get_field('gt_masks')[i]).unsqueeze(0),
                                        proposals.get_field('masks'))
      if gt_mask is not None:
        output_mask[proposals.get_field('masks')[id,0].data.cpu().numpy() == 1] = i + 1
  else:
    for i in range(len(track_ids)):
      if f == 0:
        output_mask[proposals['masks'][i,0].data.cpu().numpy() == 1] = track_ids[i]
      elif track_ids[i] in output_tracks:
        track_ids[i] = output_tracks[track_ids[i]]
        output_mask[proposals['masks'][i,0].data.cpu().numpy() == 1] = track_ids[i]
        last_update[output_tracks[track_ids[i]]] = f
      elif len(output_tracks.keys()) < 20:
        output_tracks[track_ids[i]] = len(output_tracks.keys())
        last_update = np.append(last_update,f)
        output_mask[proposals['masks'][i,0].data.cpu().numpy() == 1] = len(output_tracks.keys()) + 1


  out_file = os.path.join(out_folder, '{:05d}'.format(f) + ".png")
  save_mask(output_mask.astype(np.int), out_file)
  return output_tracks, last_update


def run_eval(line):
  print(line)
  line = line.rstrip()
  in_folder = "../results/eval_maskrcnn_warp/davis-testdev/" + line
  out_folder = "../results/eval_maskrcnn_warp/davis-testdev/updated_index/" + line
  if not os.path.exists(out_folder):
    os.makedirs(out_folder)
  all_proposals = glob.glob(os.path.join(in_folder, "*.pickle"))

  for i in range(len(all_proposals)):
    proposals_raw = pickle.load(open(all_proposals[i], 'rb'))
    if i == 0:
      output_tracks = {}
      count = 0
      last_update = np.zeros(len(proposals_raw['track_ids']))
      for track_id in proposals_raw['track_ids']:
        output_tracks[track_id] = count
        count+=1
    if USE_ORACLE_REID:
      # use gt for oracle reid
      gt_path = os.path.join(DAVIS_ROOT, "Annotations_unsupervised/480p", line, '{:05d}.png'.format(i))
      gt = get_one_hot_vectors(np.array(Image.open(gt_path), dtype=np.uint16))
      proposals_raw.add_field('gt_masks', gt)

    output_tracks, last_update = save_images_for_eval(proposals_raw, out_folder, i, output_tracks, last_update)


def main():
  seqs = root + "/ImageSets/2019/test-dev.txt"
  lines = ['bmx-trees']
  pool = mp.Pool(1)
  with open(os.path.join(seqs), "r") as lines:
   pool.map(run_eval, [line for line in lines])
  #for line in lines:
  #  run_eval(line)


if __name__ == '__main__':
  main()
