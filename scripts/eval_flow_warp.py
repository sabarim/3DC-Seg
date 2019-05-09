import glob
import os
import numpy as np

from cv2.cv2 import imread
from tqdm import tqdm

data_dir = "/globalwork/data/DAVIS-Unsupervised/DAVIS/"
flow_dir = "/globalwork/data/DAVIS-Unsupervised/DAVIS/flo/"
proposals = "/globalwork/data/DAVIS-Unsupervised/DAVIS/Annotations_Unsupervised/480p/"


def create_object_id_mapping():
  pass


def get_overlapping_proposals(frame_id, video_sequence):
  last_frame_id = frame_id - 1
  curr_frame_id = frame_id
  ann_dir = os.path.join(proposals, video_sequence)

  last_frame = imread(ann_dir + '/{:05d}.jpg'.format(last_frame_id))
  curr_frame = imread(ann_dir + '/{:05d}.jpg'.format(curr_frame_id))

  # create an object id map from the previous to the current frame
  obj_id_mapping = create_object_id_mapping(last_frame, curr_frame)

  out_img = np.zeros_like(curr_frame)
  for key in obj_id_mapping.keys():
    out_img = np.where(curr_frame == obj_id_mapping[key], key, out_img)

  return out_img


def main():
  seqs = data_dir + "ImageSets/2017/val.txt"
  with open(os.path.join(seqs), "r") as lines:
    for line in lines:
      line = line.rstrip()
      video_dir = data_dir + "JPEGImages/480p/" + line
      imgs = tqdm(list(glob.glob(video_dir + "/*.jpg")))

      # first frame does not have flow
      for i in range(1, len(list(imgs)) + 1):
        # get the proposals that has the best overlap in terms of iou
        out_ann = get_overlapping_proposals(i, line)
        im_all = [imread(img) for img in [video_dir + '/{:05d}.jpg'.format(prev_id),
                                          video_dir + '/{:05d}.jpg'.format(curr_id)]]
        im_all = [im[:, :, :3] for im in im_all]
        flow_fn = os.path.join(out_folder, '{:05d}'.format(i + 1) + ".flo")
        run_flow(im_all, out_folder, flow_fn)


if __name__ == '__main__':
  main()