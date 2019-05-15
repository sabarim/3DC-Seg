import glob
import os
import numpy as np

from cv2.cv2 import imread
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# iou overlap threshold for the objects to be considered the same
from util import save_mask

THRESH = 0.4

data_dir = "/globalwork/data/DAVIS-Unsupervised/DAVIS/"
flow_dir = "/globalwork/data/DAVIS-Unsupervised/DAVIS/flo/"
proposals = "/globalwork/data/DAVIS-Unsupervised/DAVIS/Annotations_Unsupervised/480p/"
result_folder = "results/"


def get_best_overlap(ref_obj, curr_frame):
  conf_matrix = confusion_matrix(ref_obj, curr_frame)
  conf_matrix /= conf_matrix.sum(conf_matrix, axis=0)
  return np.argmax(conf_matrix) if conf_matrix.max() > THRESH else -1


def create_object_id_mapping(last_frame, curr_frame):
  ids = np.setdiff1d(np.unique(last_frame), [0])
  result = {}
  for id in ids:
    target_id = get_best_overlap((last_frame == id).astype(np.uint8), curr_frame)
    result[id] = target_id

  return result


def get_overlapping_proposals(frame_id, video_sequence):
  # last_frame_id = frame_id - 1
  curr_frame_id = frame_id
  ann_dir = os.path.join(proposals, video_sequence)

  last_frame_warped = imread(flow_dir + 'imgs_warped/{:05d}.jpg'.format(frame_id))
  curr_frame = imread(ann_dir + '/{:05d}.jpg'.format(curr_frame_id))

  # create an object id map from the previous to the current frame
  obj_id_mapping = create_object_id_mapping(last_frame_warped, curr_frame)

  out_img = np.zeros_like(curr_frame)
  for key in obj_id_mapping.keys():
    out_img = np.where(curr_frame == obj_id_mapping[key], key, out_img)

  return out_img


def main():
  seqs = data_dir + "ImageSets/2017/val.txt"
  if not os.path.exists(result_folder):
    os.makedirs(result_folder)
  with open(os.path.join(seqs), "r") as lines:
    for line in lines:
      line = line.rstrip()
      video_dir = data_dir + "JPEGImages/480p/" + line
      imgs = tqdm(list(glob.glob(video_dir + "/*.jpg")))

      # first frame does not have flow
      for i in range(1, len(list(imgs)) + 1):
        # get the proposals that has the best overlap in terms of iou
        out_ann = get_overlapping_proposals(i, line)
        out_folder = os.path.join(result_folder, line)
        if not os.path.exists(out_folder):
          os.makedirs(out_folder)
        out_fn = os.path.join(result_folder, line, '{:05d}'.format(i + 1) + ".png")
        save_mask(out_ann, out_fn)


if __name__ == '__main__':
  main()