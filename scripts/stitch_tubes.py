import glob
import sys
import os
import torch
import numpy as np
from PIL import Image

from inference_handlers.SpatialEmbInference import OVERLAPS


DATA_DIR = ''
SAVE_PATH = ''
CLIP_LENGTH = 8


def main(source, dest, stitch):
  assert os.path.exists(source) and os.path.exists(dest)
  seqs = glob.glob(source + '/*')
  start_idx = 0
  for seq in seqs:
    seq_path = os.path.join(source, seq)
    gt_path = os.path.join(DATA_DIR, seq)
    num_frames = len(glob.glob(gt_path + "/*.png"))
    files = glob.glob(seq_path + "/*.png")
    files.sort()
    for start_idx in range(0, num_frames, CLIP_LENGTH-OVERLAPS):
      tube, tube_gt = get_tubes(gt_path, seq_path, start_idx)
      stitched_tube = stitch(tube, tube_gt)
      save_results(stitched_tube, SAVE_PATH)


def get_tubes(gt_path, seq_path, start_idx):
  tube = []
  tube_gt = []
  end_idx = start_idx + CLIP_LENGTH
  for idx in range(start_idx, end_idx):
    filename = os.path.join(seq_path, 'clip_{:05d}_{:05d}_frame_{:05d}.png'.format(start_idx, end_idx - 1, idx))
    gt_filename = os.path.join(gt_path, '{:05d}.png'.format(idx))
    tube += [torch.from_numpy(np.array(Image.open(filename), dtype=np.uint8))]
    tube_gt += [torch.from_numpy(np.array(Image.open(gt_filename), dtype=np.uint8))]
  tube = torch.stack(tube)
  tube_gt = torch.stack(tube_gt)

  return tube, tube_gt


def read_mask(file_name):
  return np.array(Image.open(file_name).convert('P'), dtype=np.uint8)


if __name__ == '__main__':
  assert len(sys.argv)  == 2
  source_path = sys.argv[0]
  result_path = sys.argv[1]

  main()