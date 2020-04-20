import glob
import os
import pickle

import numpy as np
from PIL import Image

from inference_handlers.infer_utils.Visualisation import create_color_map

maskrcnn_data_dir = "/globalwork/athar/Davis2020/maskrcnn_davis_testdev_masks/Full-Resolution/"
out_dir = "../results/davis_testdev_from_maskrcnn/"
THRESH=0.1


def main():
  # seqs = davis_data_dir + "ImageSets/2017/val.txt"
  cmap = create_color_map()
  seqs = glob.glob(maskrcnn_data_dir + "/*")
  # with open(os.path.join(seqs), "r") as lines:
  for line in seqs:
    print(line)
    line = line.rstrip()
    seq_name = line.split("/")[-1]
    out_folder = out_dir + seq_name
    if not os.path.exists(out_folder):
      os.makedirs(out_folder)
    pickle_files = glob.glob(os.path.join(maskrcnn_data_dir, line, "*.pkl"))
    davis_vid_masks = []

    for f in pickle_files:
      proposals = pickle.load(open(f, 'rb'))
      indices = proposals['scores'] > THRESH
      scores = proposals['scores'][indices]
      masks = proposals['pred_masks'][indices]
      if len(masks) > 0:
        mask_scores = np.stack([mask.byte() * score for score, mask in zip(scores, masks)], axis=0)
        bg = np.where(np.sum(mask_scores, axis=0) <= THRESH, 1, 0)
        mask_scores = np.concatenate((bg[None], mask_scores), axis=0)
        result_mask = np.argmax(mask_scores, axis=0)
        f_name = f.split("/")[-1].split(".")[0]
        out_file = os.path.join(out_folder, f_name + ".png")
        img_M = Image.fromarray(result_mask.astype(np.uint8))
        img_M.putpalette(cmap)
        img_M.save(out_file)



if __name__ == '__main__':
    main()
