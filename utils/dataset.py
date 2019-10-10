from datasets.DAVIS import DAVIS, DAVISEval, DAVISInfer
from datasets.DAVIS16 import DAVIS16Eval, DAVIS16, DAVIS16PredictOne, DAVIS16PredictOneEval, DAVIS17MaskGuidance, \
  DAVISSiam3d, DAVISSimilarity
from datasets.DAVIS3dProposalGuidance import DAVIS3dProposalGuidance, DAVIS3dProposalGuidanceEval
from datasets.DAVISProposalGuidance import DAVISProposalGuidance, DAVISProposalGuidanceEval, DAVISProposalGuidanceInfer
from datasets.YoutubeVOS import YoutubeVOSDataset, YoutubeVOSEmbedding
from utils.Constants import DAVIS_ROOT, YOUTUBEVOS_ROOT, COCO_ROOT
from datasets.static.COCO import COCODataset, COCOInstanceDataset


def get_dataset(args):
  if args.train_dataset == "davis_proposal_guidance":
    trainset = DAVISProposalGuidance(DAVIS_ROOT, imset='2017/train.txt', is_train=True,
                                     random_instance=args.random_instance, crop_size=args.crop_size, resize_mode=args.resize_mode,
                                     max_temporal_gap=12, temporal_window=args.tw, augmentors=args.augmentors,
                                     proposal_dir=args.proposal_dir)
  elif args.train_dataset == "davis16_last":
    trainset = DAVIS16PredictOne(DAVIS_ROOT, is_train=True, crop_size=args.crop_size, resize_mode=args.resize_mode,
                                 temporal_window=args.tw, augmentors=args.augmentors, proposal_dir=args.proposal_dir)
  elif args.train_dataset == "davis16_centre":
    trainset = DAVIS16PredictOne(DAVIS_ROOT, is_train=True, crop_size=args.crop_size, resize_mode=args.resize_mode,
                                 temporal_window=args.tw, augmentors=args.augmentors, proposal_dir=args.proposal_dir,
                                 predict_centre=True)
  elif args.train_dataset == "davis16":
    trainset = DAVIS16(DAVIS_ROOT, is_train=True, crop_size=args.crop_size, resize_mode=args.resize_mode,
                       temporal_window=args.tw, augmentors=args.augmentors, proposal_dir=args.proposal_dir,
                       random_instance=args.random_instance, max_temporal_gap=args.max_temporal_gap,
                       num_classes=args.n_classes)
  elif args.train_dataset == "davis_similarity":
    trainset = DAVISSimilarity(DAVIS_ROOT, is_train=True, crop_size=args.crop_size, resize_mode=args.resize_mode,
                       temporal_window=args.tw, augmentors=args.augmentors, proposal_dir=args.proposal_dir,
                       random_instance=args.random_instance, max_temporal_gap=args.max_temporal_gap,
                               num_classes=args.n_classes)
  elif args.train_dataset == "davis_siam":
    trainset = DAVISSiam3d(DAVIS_ROOT, is_train=True, crop_size=args.crop_size, resize_mode=args.resize_mode,
                       temporal_window=args.tw, augmentors=args.augmentors, proposal_dir=args.proposal_dir,
                       random_instance=args.random_instance, max_temporal_gap=args.max_temporal_gap)
  elif args.train_dataset == "davis17_mask_guidance":
    trainset = DAVIS17MaskGuidance(DAVIS_ROOT, is_train=True, crop_size=args.crop_size, resize_mode=args.resize_mode,
                       temporal_window=args.tw, augmentors=args.augmentors, proposal_dir=args.proposal_dir,
                       random_instance=args.random_instance, max_temporal_gap=args.max_temporal_gap)
  elif args.train_dataset == "davis":
    trainset = DAVIS(DAVIS_ROOT, imset='2017/train.txt', is_train=True,
                     random_instance=args.random_instance, crop_size=args.crop_size, resize_mode=args.resize_mode,
                     max_temporal_gap=12, temporal_window=args.tw, augmentors=args.augmentors,
                     proposal_dir=args.proposal_dir)
  elif args.train_dataset == "youtube_vos":
    trainset = YoutubeVOSDataset(YOUTUBEVOS_ROOT, imset='train', is_train=True,
                                 random_instance=args.random_instance, crop_size=args.crop_size,
                                 resize_mode=args.resize_mode, temporal_window=args.tw, num_classes=args.n_classes)
  elif args.train_dataset == "yvos_embedding":
    trainset = YoutubeVOSEmbedding(YOUTUBEVOS_ROOT, imset='train', is_train=True,
                                 random_instance=args.random_instance, crop_size=args.crop_size,
                                 resize_mode=args.resize_mode, temporal_window=args.tw)
  elif args.train_dataset == "davis_3d":
    trainset = DAVIS3dProposalGuidance(DAVIS_ROOT, imset='2017/train.txt', is_train=True,
                                       random_instance=args.random_instance, crop_size=args.crop_size, resize_mode=args.resize_mode,
                                       max_temporal_gap=12, temporal_window=args.tw, augmentors=args.augmentors,
                                       proposal_dir=args.proposal_dir)
  elif args.train_dataset == "coco":
    trainset = COCODataset(COCO_ROOT, is_train=True, crop_size=args.crop_size,
                                 resize_mode=args.resize_mode, temporal_window=args.tw)
  elif args.train_dataset == "coco_instance":
    trainset = COCOInstanceDataset(COCO_ROOT, is_train=True, crop_size=args.crop_size,
                           resize_mode=args.resize_mode, temporal_window=args.tw)

  if args.test_dataset == "coco":
    testset = COCODataset(COCO_ROOT, is_train=False, crop_size=args.crop_size,
                          resize_mode=args.resize_mode, temporal_window=args.tw)
  elif args.test_dataset == "coco_instance":
    testset = COCOInstanceDataset(COCO_ROOT, is_train=False, crop_size=args.crop_size,
                          resize_mode=args.resize_mode, temporal_window=args.tw)
  elif 'davis_proposal_guidance' in args.test_dataset:
    testset = DAVISProposalGuidanceEval(DAVIS_ROOT, imset='2017/val.txt', random_instance=args.random_instance,
                                        crop_size=args.crop_size_eval, resize_mode=args.resize_mode_eval,
                                        temporal_window=args.tw, proposal_dir=args.proposal_dir)
  elif "davis16_last" in args.test_dataset:
    testset = DAVIS16PredictOneEval(DAVIS_ROOT, crop_size=args.crop_size_eval,
                                    resize_mode=args.resize_mode_eval, temporal_window=args.tw, proposal_dir=args.proposal_dir)
  elif "davis16_centre" in args.test_dataset:
    testset = DAVIS16PredictOneEval(DAVIS_ROOT, crop_size=args.crop_size_eval,
                                    resize_mode=args.resize_mode_eval, temporal_window=args.tw, proposal_dir=args.proposal_dir,
                                    predict_centre = True)
  elif "davis16" in args.test_dataset:
    testset = DAVIS16Eval(DAVIS_ROOT, crop_size=args.crop_size_eval, random_instance=args.random_instance,
                          resize_mode=args.resize_mode_eval, temporal_window=args.tw, proposal_dir=args.proposal_dir,
                          num_classes=args.n_classes)
  elif "davis_similarity" in args.test_dataset:
    testset = DAVISSimilarity(DAVIS_ROOT, imset="2017/val.txt", crop_size=args.crop_size_eval, random_instance=args.random_instance,
                          resize_mode=args.resize_mode_eval, temporal_window=args.tw, proposal_dir=args.proposal_dir,
                              num_classes=args.n_classes)
  elif "davis_siam" in args.test_dataset:
    testset = DAVISSiam3d(DAVIS_ROOT, crop_size=args.crop_size_eval, random_instance=args.random_instance,
                          resize_mode=args.resize_mode_eval, temporal_window=args.tw, proposal_dir=args.proposal_dir)
  elif "davis17_3d" in args.test_dataset:
    testset = DAVIS16(DAVIS_ROOT, imset="2017/val.txt", is_train=False, crop_size=args.crop_size, resize_mode=args.resize_mode,
                       temporal_window=args.tw, augmentors=args.augmentors, proposal_dir=args.proposal_dir,
                       random_instance=args.random_instance)
  elif "davis17_mask_guidance" in args.test_dataset:
    testset = DAVIS17MaskGuidance(DAVIS_ROOT, imset="2017/val.txt", is_train=False, crop_size=args.crop_size, resize_mode=args.resize_mode,
                       temporal_window=args.tw, augmentors=args.augmentors, proposal_dir=args.proposal_dir,
                       random_instance=args.random_instance)
  elif 'davis_3d' in args.test_dataset:
    testset = DAVIS3dProposalGuidanceEval(DAVIS_ROOT, imset='2017/val.txt', random_instance=False, crop_size=args.crop_size_eval,
                                          resize_mode=args.resize_mode_eval, temporal_window=args.tw, proposal_dir=args.proposal_dir)
  elif 'davis' in args.test_dataset:
    testset = DAVISEval(DAVIS_ROOT, imset='2017/val.txt', random_instance=False, crop_size=args.crop_size_eval,
                        resize_mode=args.resize_mode_eval, temporal_window=args.tw, proposal_dir=args.proposal_dir)
  elif args.test_dataset == "youtube_vos":
    testset = YoutubeVOSDataset(YOUTUBEVOS_ROOT, imset='valid', is_train=False,
                                 random_instance=args.random_instance, crop_size=args.crop_size,
                                 resize_mode=args.resize_mode, temporal_window=args.tw, num_classes=args.n_classes)

  if 'infer' in args.task:
    if 'davis_proposal_guidance' in args.test_dataset:
      testset = DAVISProposalGuidanceInfer(DAVIS_ROOT, random_instance=False, crop_size=args.crop_size_eval,
                                           resize_mode=args.resize_mode_eval, temporal_window=args.tw, proposal_dir=args.proposal_dir)
    elif 'davis_infer'in args.test_dataset:
      testset = DAVISInfer(DAVIS_ROOT, random_instance=False, crop_size=args.crop_size_eval,
                           resize_mode=args.resize_mode_eval, temporal_window=args.tw, proposal_dir=args.proposal_dir)

  return trainset, testset
