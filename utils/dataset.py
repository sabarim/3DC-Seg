from utils.Constants import DAVIS_ROOT, YOUTUBEVOS_ROOT, COCO_ROOT, YOUTUBEVIS_ROOT, MAPILLARY_ROOT


def get_dataset(args):
  if args.train_dataset == "davis_proposal_guidance":
    from datasets.DAVISProposalGuidance import DAVISProposalGuidance
    trainset = DAVISProposalGuidance(DAVIS_ROOT, imset='2017/train.txt', is_train=True,
                                     random_instance=args.random_instance, crop_size=args.crop_size, resize_mode=args.resize_mode,
                                     max_temporal_gap=12, temporal_window=args.tw, augmentors=args.augmentors,
                                     proposal_dir=args.proposal_dir)
  elif args.train_dataset == "davis16_last":
    from datasets.DAVIS16 import DAVIS16PredictOne
    trainset = DAVIS16PredictOne(DAVIS_ROOT, is_train=True, crop_size=args.crop_size, resize_mode=args.resize_mode,
                                 temporal_window=args.tw, augmentors=args.augmentors, proposal_dir=args.proposal_dir)
  elif args.train_dataset == "davis16_centre":
    from datasets.DAVIS16 import DAVIS16PredictOne
    trainset = DAVIS16PredictOne(DAVIS_ROOT, is_train=True, crop_size=args.crop_size, resize_mode=args.resize_mode,
                                 temporal_window=args.tw, augmentors=args.augmentors, proposal_dir=args.proposal_dir,
                                 predict_centre=True)
  elif args.train_dataset == "davis16":
    from datasets.DAVIS16 import DAVIS16
    trainset = DAVIS16(DAVIS_ROOT, is_train=True, crop_size=args.crop_size, resize_mode=args.resize_mode,
                       temporal_window=args.tw, augmentors=args.augmentors, proposal_dir=args.proposal_dir,
                       random_instance=args.random_instance, max_temporal_gap=args.max_temporal_gap,
                       num_classes=args.n_classes)
  elif args.train_dataset == "davis_similarity":
    from datasets.DAVIS16 import DAVISSimilarity
    trainset = DAVISSimilarity(DAVIS_ROOT, is_train=True, crop_size=args.crop_size, resize_mode=args.resize_mode,
                       temporal_window=args.tw, augmentors=args.augmentors, proposal_dir=args.proposal_dir,
                       random_instance=args.random_instance, max_temporal_gap=args.max_temporal_gap,
                               num_classes=args.n_classes)
  elif args.train_dataset == "davis_siam":
    from datasets.DAVIS16 import DAVISSiam3d
    trainset = DAVISSiam3d(DAVIS_ROOT, is_train=True, crop_size=args.crop_size, resize_mode=args.resize_mode,
                       temporal_window=args.tw, augmentors=args.augmentors, proposal_dir=args.proposal_dir,
                       random_instance=args.random_instance, max_temporal_gap=args.max_temporal_gap)
  elif args.train_dataset == "davis17_mask_guidance":
    from datasets.DAVIS16 import DAVIS17MaskGuidance
    trainset = DAVIS17MaskGuidance(DAVIS_ROOT, is_train=True, crop_size=args.crop_size, resize_mode=args.resize_mode,
                       temporal_window=args.tw, augmentors=args.augmentors, proposal_dir=args.proposal_dir,
                       random_instance=args.random_instance, max_temporal_gap=args.max_temporal_gap)
  elif args.train_dataset == "davis":
    from datasets.DAVIS import DAVIS
    trainset = DAVIS(DAVIS_ROOT, imset='2017/train.txt', is_train=True,
                     random_instance=args.random_instance, crop_size=args.crop_size, resize_mode=args.resize_mode,
                     max_temporal_gap=12, temporal_window=args.tw, augmentors=args.augmentors,
                     proposal_dir=args.proposal_dir)
  elif args.train_dataset == "youtube_vos":
    from datasets.YoutubeVOS import YoutubeVOSDataset
    trainset = YoutubeVOSDataset(YOUTUBEVOS_ROOT, imset='train', is_train=True,
                                 random_instance=args.random_instance, crop_size=args.crop_size,
                                 resize_mode=args.resize_mode, temporal_window=args.tw, num_classes=args.n_classes)
  elif args.train_dataset == "yvos_embedding":
    from datasets.YoutubeVOS import YoutubeVOSEmbedding
    trainset = YoutubeVOSEmbedding(YOUTUBEVOS_ROOT, imset='train', is_train=True,
                                 random_instance=args.random_instance, crop_size=args.crop_size,
                                 resize_mode=args.resize_mode, temporal_window=args.tw)
  elif args.train_dataset == "youtube_vis":
    from datasets.YoutubeVIS import YoutubeVISDataset
    trainset = YoutubeVISDataset(YOUTUBEVIS_ROOT, imset='train', is_train=True,
                                 random_instance=args.random_instance, crop_size=args.crop_size,
                                 resize_mode=args.resize_mode, temporal_window=args.tw)
  elif args.train_dataset == "davis_3d":
    from datasets.DAVIS3dProposalGuidance import DAVIS3dProposalGuidance
    trainset = DAVIS3dProposalGuidance(DAVIS_ROOT, imset='2017/train.txt', is_train=True,
                                       random_instance=args.random_instance, crop_size=args.crop_size, resize_mode=args.resize_mode,
                                       max_temporal_gap=12, temporal_window=args.tw, augmentors=args.augmentors,
                                       proposal_dir=args.proposal_dir)
  elif args.train_dataset == "coco":
    from datasets.coco.COCO import COCODataset
    trainset = COCODataset(COCO_ROOT, is_train=True, crop_size=args.crop_size,
                                 resize_mode=args.resize_mode, temporal_window=args.tw)
  elif args.train_dataset == "coco_instance":
    from datasets.coco.COCO import COCOInstanceDataset
    trainset = COCOInstanceDataset(COCO_ROOT, is_train=True, crop_size=args.crop_size,
                           resize_mode=args.resize_mode, temporal_window=args.tw)
  elif args.train_dataset == "coco_embedding":
    from datasets.coco.COCO import COCOEmbeddingDataset
    trainset = COCOEmbeddingDataset(COCO_ROOT, is_train=True, crop_size=args.crop_size,
                           resize_mode=args.resize_mode, temporal_window=args.tw)
  elif args.train_dataset == "mapillary":
    from datasets.mapillary.MapillaryInstance import MapillaryVideoDataset
    trainset = MapillaryVideoDataset(MAPILLARY_ROOT, is_train=True, crop_size=args.crop_size,
                                 resize_mode=args.resize_mode, temporal_window=args.tw, min_size=args.min_size)
  elif args.train_dataset == "coco_mapillary":
    from datasets.mapillary.COCOMapillary import COCOMapillary
    trainset = COCOMapillary(is_train=True, crop_size=args.crop_size,
                             resize_mode=args.resize_mode, temporal_window=args.tw, min_size=args.min_size)

  # Validation dataset
  if args.test_dataset == "coco":
    from datasets.coco.COCO import COCODataset
    testset = COCODataset(COCO_ROOT, is_train=False, crop_size=args.crop_size,
                          resize_mode=args.resize_mode, temporal_window=args.tw)
  elif args.test_dataset == "coco_instance":
    from datasets.coco.COCO import COCOInstanceDataset
    testset = COCOInstanceDataset(COCO_ROOT, is_train=False, crop_size=args.crop_size,
                          resize_mode=args.resize_mode, temporal_window=args.tw)
  elif args.test_dataset == "coco_embedding":
    from datasets.coco.COCO import COCOEmbeddingDataset
    testset = COCOEmbeddingDataset(COCO_ROOT, is_train=False, crop_size=args.crop_size,
                                  resize_mode=args.resize_mode, temporal_window=args.tw)
  elif args.test_dataset == "mapillary":
    from datasets.mapillary.MapillaryInstance import MapillaryVideoDataset
    testset = MapillaryVideoDataset(MAPILLARY_ROOT, is_train=False, crop_size=args.crop_size_eval,
                                 resize_mode=args.resize_mode_eval, temporal_window=args.tw, min_size=args.min_size)
  elif 'davis_proposal_guidance' in args.test_dataset:
    from datasets.DAVISProposalGuidance import DAVISProposalGuidanceEval
    testset = DAVISProposalGuidanceEval(DAVIS_ROOT, imset='2017/val.txt', random_instance=args.random_instance,
                                        crop_size=args.crop_size_eval, resize_mode=args.resize_mode_eval,
                                        temporal_window=args.tw, proposal_dir=args.proposal_dir)
  elif "davis16_last" in args.test_dataset:
    from datasets.DAVIS16 import DAVIS16PredictOneEval
    testset = DAVIS16PredictOneEval(DAVIS_ROOT, crop_size=args.crop_size_eval,
                                    resize_mode=args.resize_mode_eval, temporal_window=args.tw, proposal_dir=args.proposal_dir)
  elif "davis16" in args.test_dataset:
    from datasets.DAVIS16 import DAVIS16Eval
    testset = DAVIS16Eval(DAVIS_ROOT, crop_size=args.crop_size_eval, random_instance=args.random_instance,
                          resize_mode=args.resize_mode_eval, temporal_window=args.tw, proposal_dir=args.proposal_dir,
                          num_classes=args.n_classes)
  elif "davis_similarity" in args.test_dataset:
    from datasets.DAVIS16 import DAVISSimilarity
    testset = DAVISSimilarity(DAVIS_ROOT, imset="2017/val.txt", crop_size=args.crop_size_eval, random_instance=args.random_instance,
                          resize_mode=args.resize_mode_eval, temporal_window=args.tw, proposal_dir=args.proposal_dir,
                              num_classes=args.n_classes)
  elif "davis_siam" in args.test_dataset:
    from datasets.DAVIS16 import DAVISSiam3d
    testset = DAVISSiam3d(DAVIS_ROOT, crop_size=args.crop_size_eval, random_instance=args.random_instance,
                          resize_mode=args.resize_mode_eval, temporal_window=args.tw, proposal_dir=args.proposal_dir)
  elif "davis17_3d" in args.test_dataset:
    from datasets.DAVIS16 import DAVIS16
    testset = DAVIS16(DAVIS_ROOT, imset="2017/val.txt", is_train=False, crop_size=args.crop_size, resize_mode=args.resize_mode,
                       temporal_window=args.tw, augmentors=args.augmentors, proposal_dir=args.proposal_dir,
                       random_instance=args.random_instance)
  elif "davis17_mask_guidance" in args.test_dataset:
    from datasets.DAVIS16 import DAVIS17MaskGuidance
    testset = DAVIS17MaskGuidance(DAVIS_ROOT, imset="2017/val.txt", is_train=False, crop_size=args.crop_size, resize_mode=args.resize_mode,
                       temporal_window=args.tw, augmentors=args.augmentors, proposal_dir=args.proposal_dir,
                       random_instance=args.random_instance)
  elif 'davis_3d' in args.test_dataset:
    from datasets.DAVIS3dProposalGuidance import DAVIS3dProposalGuidanceEval
    testset = DAVIS3dProposalGuidanceEval(DAVIS_ROOT, imset='2017/val.txt', random_instance=False, crop_size=args.crop_size_eval,
                                          resize_mode=args.resize_mode_eval, temporal_window=args.tw, proposal_dir=args.proposal_dir)
  elif 'davis' in args.test_dataset:
    from datasets.DAVIS import DAVISEval
    testset = DAVISEval(DAVIS_ROOT, imset='2017/val.txt', random_instance=False, crop_size=args.crop_size_eval,
                        resize_mode=args.resize_mode_eval, temporal_window=args.tw, proposal_dir=args.proposal_dir)
  elif args.test_dataset == "youtube_vos":
    from datasets.YoutubeVOS import YoutubeVOSDataset
    testset = YoutubeVOSDataset(YOUTUBEVOS_ROOT, imset='valid', is_train=False,
                                 random_instance=args.random_instance, crop_size=args.crop_size,
                                 resize_mode=args.resize_mode, temporal_window=args.tw, num_classes=args.n_classes)
  elif args.test_dataset == "youtube_vis":
    from datasets.YoutubeVIS import YoutubeVISDataset
    testset = YoutubeVISDataset(YOUTUBEVIS_ROOT, imset='valid', is_train=False,
                                 random_instance=False, crop_size=args.crop_size,
                                 resize_mode=args.resize_mode, temporal_window=args.tw, num_classes=args.n_classes)

  if 'infer' in args.task:
    if 'davis_proposal_guidance' in args.test_dataset:
      from datasets.DAVISProposalGuidance import DAVISProposalGuidanceInfer
      testset = DAVISProposalGuidanceInfer(DAVIS_ROOT, random_instance=False, crop_size=args.crop_size_eval,
                                           resize_mode=args.resize_mode_eval, temporal_window=args.tw, proposal_dir=args.proposal_dir)
    elif 'davis_infer'in args.test_dataset:
      from datasets.DAVIS import DAVISInfer
      testset = DAVISInfer(DAVIS_ROOT, random_instance=False, crop_size=args.crop_size_eval,
                           resize_mode=args.resize_mode_eval, temporal_window=args.tw, proposal_dir=args.proposal_dir)

  return trainset, testset
