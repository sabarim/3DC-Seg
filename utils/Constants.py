DAVIS_ROOT = '/globalwork/data/DAVIS-Unsupervised/DAVIS/'
YOUTUBEVOS_ROOT = '/globalwork/data/youtube-vos/'
YOUTUBEVIS_ROOT = '/globalwork/data/YouTube-VOS-VIS/'
COCO_ROOT = '/globalwork/mahadevan/mywork/data/coco/'
MAPILLARY_ROOT = "/globalwork/voigtlaender/data/mapillary/"
KITTI_ROOT = "/globalwork/data/KITTI_MOTS/"
FBMS_ROOT = "/globalwork/data/fbms/"
VISAL_ROOT = "/globalwork/mahadevan/mywork/data/ViSal/"
SEGTRACK_ROOT = "/globalwork/data/segtrack-v2/"
# network_models = {0:"RGMP", 1:"FeatureAgg3d", 2: "FeatureAgg3dMergeTemporal", 3: "FeatureAgg3dMulti",
#                   4: "FeatureAgg3dMulti101", 5: "Resnet3d", 6: "Resnet3dPredictOne", 7: "Resnet3dMaskGuidance",
#                   8: "SiamResnet3d", 9:"Resnet3dNonLocal", 10: "Resnet3dSimilarity", 11:"Resnet3dEmbeddingNetwork",
#                   12: "Resnet3dSegmentEmbedding", 13: "Resnet3dSpatialEmbedding", 14: "Resnet3dEmbeddingMultiDecoder",
#                   15: "Resnet3dChannelSeparated_ip", 16: "Resnet3dChannelSeparated_ir", 17: "Resnet3dCSNiRSameDecoders",
#                   18: "Resnet3dCSNiRMultiScale", 19: "Resnet3dCSNiRMultiClass", 20: "Resnet3dCSNiRLight",
#                   21: "Resnet3d101", 22:"ResnetCSNNoGC", 23: "ResnetCSNNonLocal", 24: "ResnetCSN", 25: "R2plus1d"}
network_models = {0: "Resnet3d101", 1: "ResnetCSN", 25: "R2plus1d", 3: "ResnetCSNNoGC", 4: "ResnetCSNNonLocal"}
#DAVIS_ROOT = '/disk2/data/DAVIS/'
MODEL_ROOT = '/globalwork/mahadevan/vision/davis-unsupervised/saved_models/'

# Optimisers
ADAM_OPTIMISER = "Adam"

PALETTE = [
  0, 0, 0,
  31, 119, 180,
  174, 199, 232,
  255, 127, 14,
  255, 187, 120,
  44, 160, 44,
  152, 223, 138,
  214, 39, 40,
  255, 152, 150,
  148, 103, 189,
  197, 176, 213,
  140, 86, 75,
  196, 156, 148,
  227, 119, 194,
  247, 182, 210,
  127, 127, 127,
  199, 199, 199,
  188, 189, 34,
  219, 219, 141,
  23, 190, 207,
  158, 218, 229
]


class font:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


PRED_SEM_SEG = 'sem_seg'
PRED_EMBEDDING = 'embedding'
PRED_LOGITS = 'logits'

