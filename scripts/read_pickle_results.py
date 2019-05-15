import glob
import os
import pickle


RESULT_ROOT = "../results/eval_maskrcnn_warp/"
DAVIS_ROOT = '/globalwork/data/DAVIS-Unsupervised/DAVIS/'

def read_results():
  imset = os.path.join(DAVIS_ROOT, 'ImageSets', '2017', 'val.txt')
  results = {}
  with open(imset) as seqs:
    # for each davis video sequence
    for line in seqs:
      seq = line.rstrip()
      sequence_results = {}
      seq_results_dir = os.path.join(RESULT_ROOT, seq)
      pickle_files = glob.glob(seq_results_dir + "/*.pickle")
      for pickle_file in pickle_files:
        # each pickle file contains an object of class 'strctures.bounding_box.BoxList'
        result = pickle.load(open(pickle_file, 'rb'))
        # get the segmentation mask stored as a torch tensor of size P*1*H*W
        # where 'P' is the number of proposals for frame 't'
        masks = result.get_field('mask')
        # list of size 'P' track ids correspond to the segmentation masks
        # for eg: track_ids[0] gives the track id for proposal masks[0]
        track_ids = result.get_field('track_ids')
        # iou score corresponds to the matching scores used to associate tracks.
        # this is again a list of size 'P' ordered by the proposals
        iou_scores = result.get_field('ious')
        # get the maskrcnn proposal scores for each proposal
        maskrcnn_score = result.get_field('scores')

        sequence_results[os.path.splitext(os.path.basename(pickle_file))[0]] = \
          {'masks': masks, 'track_ids': track_ids, 'iou_scores':iou_scores, 'maskrcnn_score': maskrcnn_score}

      results[seq] = sequence_results

  return results


if __name__ == '__main__':
    results = read_results()
    print("succesfully read all results")