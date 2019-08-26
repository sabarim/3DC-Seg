import os

import pickle
import numpy as np
from datasets.DAVIS import DAVIS
from util import top_n_predictions_maskrcnn
from utils.Resize import ResizeMode


class DAVISProposalGuidance(DAVIS):
  def __init__(self, root, imset='2017/train.txt', resolution='480p', is_train=False,
               random_instance=False, num_classes=2, crop_size=None,temporal_window=5, min_temporal_gap=2,
               max_temporal_gap=8, resize_mode=ResizeMode.FIXED_SIZE, proposal_dir=None, augmentors=None):
    super(DAVISProposalGuidance, self).__init__(root, imset, resolution, is_train,
               random_instance, num_classes, crop_size,temporal_window, min_temporal_gap,
               max_temporal_gap, resize_mode, proposal_dir, augmentors)

  def read_proposals(self, video, f, gt_mask):
    proposal_file = os.path.join(self.proposal_dir, video, '{:05d}.pickle'.format(f))
    proposals = pickle.load(open(proposal_file, 'rb'))
    proposals = top_n_predictions_maskrcnn(proposals, self.max_proposals)
    proposals, object_mapping = self.filter_proposals(gt_mask, proposals)

    raw_proposals = np.zeros((self.max_proposals,) + tuple(gt_mask.shape))
    proposal_categories = np.zeros(self.max_proposals)
    proposal_scores = np.zeros(self.max_proposals)
    if len(proposals['mask']) > 0:
      raw_proposals[:len(proposals['mask'])] = proposals['mask'][:, 0].data.cpu().numpy()
    else:
      print("WARN: no proposals found in {}".format(proposal_file))
    num_proposals = len(proposals['mask'])
    proposal_scores[:len(proposals['mask'])] = proposals['scores']
    proposal_categories[:len(proposals['mask'])] = proposals['labels']
    proposal_mask = self.get_proposal_maks(proposals['scores'], proposals['mask'])

    return num_proposals, raw_proposals, proposal_mask, proposal_scores, proposal_categories

  def __getitem__(self, item):
    input_dict = super(DAVISProposalGuidance, self).__getitem__(item)
    input_dict['masks_guidance'][:, 1:-1] = input_dict['proposals'][:,1:-1]
    return input_dict