import os
import glob
import pickle

import cv2
from cv2.cv2 import imread
from scipy.misc import imshow

from util import overlay_predicitons

maskrcnn_data_dir = "/globalwork/mahadevan/mywork/data/training/pytorch/forwarded/maskrcnn/"
davis_data_dir = '/globalwork/data/DAVIS-Unsupervised/DAVIS/'
flow_dir = "/globalwork/data/DAVIS-Unsupervised/DAVIS/flo/"
out_dir = "../results/maskrcnn_warped/"


def main():
  # seqs = davis_data_dir + "ImageSets/2017/val.txt"
  seqs = ['bike-packing']
  # with open(os.path.join(seqs), "r") as lines:
  for line in seqs:
    print(line)
    line = line.rstrip()
    out_folder = out_dir + line
    if not os.path.exists(out_folder):
      os.makedirs(out_folder)
    pickle_files = glob.glob(os.path.join(maskrcnn_data_dir, line, "*.pickle"))

    for i in range(len(pickle_files) - 1):
      flow_fn = os.path.join(flow_dir, line, '{:05d}'.format(i + 1) + ".flo")
      proposals = pickle.load(open(os.path.join(maskrcnn_data_dir, line, '{:05d}'.format(i) + ".pickle"), 'rb'))
      image = imread(os.path.join(davis_data_dir, 'JPEGImages/480p/',line, '{:05d}'.format(i) + ".jpg"))
      overlayed_image = overlay_predicitons(image, proposals)
      imshow(overlayed_image)


if __name__ == '__main__':
    main()