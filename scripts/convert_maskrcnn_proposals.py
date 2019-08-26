import glob
import os
import pickle
import multiprocessing as mp
from scripts.path_constants import DAVIS_ROOT, MASKRCNN_PROPOSALS

"""
Convert maskrcnn proposals from BoxList data structure to a simple dictionary
"""

OUT_DIR = "../results/converted_proposals/thresh-0-all_fields/"
seqs = DAVIS_ROOT + "ImageSets/2017/val.txt"
# seqs = '/globalwork/data/DAVIS-Unsupervised//DAVIS-test/DAVIS/' + 'ImageSets/2019/test-challenge.txt'
# seqs = '/globalwork/data/DAVIS-Unsupervised/DAVIS-2019-Unsupervised-test-dev-480p/DAVIS/' + 'ImageSets/2019/test-dev.txt'


def convert(video):
  print(video)
  line = video.rstrip()
  out_folder = OUT_DIR + line
  if not os.path.exists(out_folder):
    os.makedirs(out_folder)
  all_proposals = glob.glob(os.path.join(MASKRCNN_PROPOSALS, "thresh-0", line, "*.pickle"))

  for i in range(len(all_proposals)):
    proposals_raw_path = os.path.join(MASKRCNN_PROPOSALS, "thresh-0", line, '{:05d}'.format(i) + ".pickle")
    proposals_raw = pickle.load(open(proposals_raw_path, 'rb'))
    result_dict = {}
    for field in proposals_raw.fields():
      result_dict.update({field: proposals_raw.get_field(field)})
      # result_dict = {"masks":proposals_raw.get_field('mask'), "scores": proposals_raw.get_field('scores')}
    pickle.dump(result_dict, open(out_folder + '/{:05d}'.format(i) + ".pickle", 'wb'))



def main():
  lines = ['drift-straight']
  pool = mp.Pool(1)
  with open(os.path.join(seqs), "r") as lines:
   pool.map(convert, [line for line in lines])
  # for line in lines:
  #   convert(line)


if __name__ == '__main__':
  main()
