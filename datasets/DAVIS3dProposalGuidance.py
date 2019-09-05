from datasets.DAVISProposalGuidance import DAVISProposalGuidance, DAVISProposalGuidanceEval
from utils.Resize import ResizeMode


class DAVIS3dProposalGuidance(DAVISProposalGuidance):
  def __init__(self, root, imset='2017/train.txt', resolution='480p', is_train=False,
               random_instance=False, num_classes=2, crop_size=None,temporal_window=5, min_temporal_gap=2,
               max_temporal_gap=8, resize_mode=ResizeMode.FIXED_SIZE, proposal_dir=None, augmentors=None):
    super(DAVIS3dProposalGuidance, self).__init__(root, imset, resolution, is_train,
                                                  random_instance, num_classes, crop_size, temporal_window, min_temporal_gap,
                                                  max_temporal_gap, resize_mode, proposal_dir, augmentors)

  def __getitem__(self, item):
    input_dict = super(DAVIS3dProposalGuidance, self).__getitem__(item)
    input_dict['masks_guidance'][:, 1:] = input_dict['proposals'][:,1:]
    input_dict['target'] = input_dict['raw_masks']
    return input_dict


class DAVIS3dProposalGuidanceEval(DAVISProposalGuidanceEval):
  def __init__(self, root, imset='2017/train.txt', resolution='480p', is_train=False,
               random_instance=False, num_classes=2, crop_size=None,temporal_window=5, min_temporal_gap=2,
               max_temporal_gap=8, resize_mode=ResizeMode.FIXED_SIZE, proposal_dir=None, augmentors=None):
    super(DAVIS3dProposalGuidanceEval, self).__init__(root, imset, is_train=is_train, crop_size=crop_size,
                                                      temporal_window=temporal_window, random_instance=random_instance,
                                                      resize_mode=resize_mode, proposal_dir=proposal_dir)

  def __getitem__(self, item):
    input_dict = super(DAVIS3dProposalGuidanceEval, self).__getitem__(item)
    input_dict['masks_guidance'][:, 1:] = input_dict['proposals'][:,1:]
    input_dict['target'] = input_dict['raw_masks']
    return input_dict