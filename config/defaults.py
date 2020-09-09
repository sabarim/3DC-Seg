from .config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
_C.VERSION = 2
_C.NAME = ""

_C.MODEL = CN()
_C.MODEL.MASK_ON = False
_C.MODEL.NETWORK = "ResnetCSN"
_C.MODEL.PRETRAINED = False
_C.MODEL.N_CLASSES = 2
_C.MODEL.FREEZE_BN = False
_C.MODEL.WEIGHTS = ""

_C.MODEL.DECODER = CN()
_C.MODEL.DECODER.INTER_BLOCK = "GC3d"
_C.MODEL.DECODER.REFINE_BLOCK = "Refine3d"

# Values to be used for image normalization (RGB order, since INPUT.FORMAT defaults to RGB).
# ImageNet: [103.530, 116.280, 123.675]
_C.MODEL.PIXEL_MEAN = [114.7748, 107.7354, 99.4750]
# ImageNet: [57.375, 57.120, 58.395]
_C.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the smallest side of the image during training
_C.INPUT.RESIZE_MODE_TRAIN = "unchanged"
_C.INPUT.RESIZE_SHAPE_TRAIN = ()

_C.INPUT.RESIZE_MODE_TEST = "unchanged"
_C.INPUT.RESIZE_SHAPE_TEST = ()
# temporal window size of input
_C.INPUT.TW = 8

# Whether the model needs RGB, YUV, HSV etc.
# Should be one of the modes defined here, as we use PIL to read the image:
# https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
# with BGR being the one exception. One can set image format to BGR, we will
# internally use RGB for conversion and flip the channels over
_C.INPUT.FORMAT = "RGB"
# The ground truth mask format that the model will use.
# Mask R-CNN supports either "polygon" or "bitmask" as ground truth.
_C.INPUT.MASK_FORMAT = "polygon"  # alternative: "bitmask"

_C.TRAINING = CN()
_C.TRAINING.BATCH_SIZE = 1
_C.TRAINING.OPTIMISER = "Adam"
_C.TRAINING.BASE_LR = 0.0001
_C.TRAINING.STEPS = (60000, 80000)
_C.TRAINING.MAX_ITER = 90000
_C.TRAINING.NUM_EPOCHS = 100

_C.TRAINING.LOSSES = CN()
_C.TRAINING.LOSSES.NAME = ["ce"]
_C.TRAINING.LOSSES.USE_IGNORE_MASK = False
_C.TRAINING.LOSSES.MULTI_CLASS = False
_C.TRAINING.LOSSES.BOOTSTRAP = False
_C.TRAINING.LOSSES.MASK_CONSISTENCY = False
_C.TRAINING.LOSSES.CRITERION = ""
_C.TRAINING.LOSSES.SCALE = []
_C.TRAINING.PRECISION = "fp32"
_C.TRAINING.LR_SCHEDULERS = []
_C.TRAINING.EVAL_EPOCH = 1

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training. Must be registered in DatasetCatalog
_C.DATASETS.TRAIN = ""
_C.DATASETS.TRAIN_ROOT = ""
# List of the pre-computed proposal files for training, which must be consistent
# with datasets listed in DATASETS.TRAIN.
_C.DATASETS.PROPOSAL_FILES_TRAIN = ()
# Number of top scoring precomputed proposals to keep for training
_C.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN = 2000
# List of the dataset names for testing. Must be registered in DatasetCatalog
_C.DATASETS.TEST = ""
_C.DATASETS.TEST_ROOT = ""
# dataset parameters
_C.DATASETS.RANDOM_INSTANCE = False
# max temporal gap to use while sampling input frames
_C.DATASETS.MAX_TEMPORAL_GAP = 8

# DAVIS parameters
_C.DATASETS.IMSET = "2017/val.txt"


# -----------------------------------------------------------------------------
# Inference settings
# -----------------------------------------------------------------------------
_C.INFERENCE = CN()
_C.INFERENCE.ENGINE = ""
_C.INFERENCE.EXHAUSTIVE = False
_C.INFERENCE.CLIP_OVERLAP = 3
_C.INFERENCE.SAVE_LOGITS = False


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4
# If True, each batch should contain only images for which the aspect ratio
# is compatible. This groups portrait images together, and landscape images
# are not batched with portrait images.
_C.DATALOADER.ASPECT_RATIO_GROUPING = True
# Options: TrainingSampler, RepeatFactorTrainingSampler
_C.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"
# Repeat threshold for RepeatFactorTrainingSampler
_C.DATALOADER.REPEAT_THRESHOLD = 0.0
# if True, the dataloader will filter out images that have no associated
# annotations at train time.
_C.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
_C.DATALOADER.NUM_SAMPLES = -1

# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()

_C.MODEL.BACKBONE.NAME = "build_resnet_backbone"
# Freeze the first several stages so they are not trained.
# There are 5 stages in ResNet. The first is a convolution, and the following
# stages are each group of residual blocks.
_C.MODEL.BACKBONE.FREEZE_AT = 2
_C.MODEL.BACKBONE.PRETRAINED_WTS = ""
_C.MODEL.BACKBONE.FREEZE_BN = False


# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()

_C.SOLVER.LR_SCHEDULERS = ["WarmupMultiStepLR"]

_C.SOLVER.MAX_ITER = 40000

_C.SOLVER.BASE_LR = 0.001

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.NESTEROV = False

_C.SOLVER.WEIGHT_DECAY = 0.0001
# The weight decay that's applied to parameters of normalization layers
# (typically the affine transformation)
_C.SOLVER.WEIGHT_DECAY_NORM = 0.0

_C.SOLVER.GAMMA = 0.1
# The epoch number to decrease learning rate by GAMMA.
_C.SOLVER.STEPS = [10]

_C.SOLVER.WARMUP_FACTOR = 1.0 / 1000
_C.SOLVER.WARMUP_ITERS = 1000
_C.SOLVER.WARMUP_METHOD = "linear"

# Save a checkpoint after every this number of iterations
_C.SOLVER.CHECKPOINT_PERIOD = 5000

# Number of images per batch across all machines.
# If we have 16 GPUs and IMS_PER_BATCH = 32,
# each GPU will see 2 images per batch.
_C.SOLVER.IMS_PER_BATCH = 16

# Detectron v1 (and previous detection code) used a 2x higher LR and 0 WD for
# biases. This is not useful (at least for recent models). You should avoid
# changing these and they exist only to reproduce Detectron v1 training if
# desired.
_C.SOLVER.BIAS_LR_FACTOR = 1.0
_C.SOLVER.WEIGHT_DECAY_BIAS = _C.SOLVER.WEIGHT_DECAY

# Gradient clipping
_C.SOLVER.CLIP_GRADIENTS = CN({"ENABLED": False})
# Type of gradient clipping, currently 2 values are supported:
# - "value": the absolute values of elements of each gradients are clipped
# - "norm": the norm of the gradient for each parameter is clipped thus
#   affecting all elements in the parameter
_C.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
# Maximum absolute value used for clipping gradients
_C.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
# Floating point number p for L-p norm to be used with the "norm"
# gradient clipping type; for L-inf, please specify .inf
_C.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Directory where output files are written
_C.OUTPUT_DIR = "./output"
# Set seed to negative to fully randomize everything.
# Set seed to positive to use a fixed seed. Note that a fixed seed increases
# reproducibility but does not guarantee fully deterministic behavior.
# Disabling all parallelism further increases reproducibility.
_C.SEED = -1
# Benchmark different cudnn algorithms.
# If input images have very different sizes, this option will have large overhead
# for about 10k iterations. It usually hurts total time, but can benefit for certain models.
# If input images have the same or similar sizes, benchmark is often helpful.
_C.CUDNN_BENCHMARK = False
# The period (in terms of steps) for minibatch visualization at train time.
# Set to 0 to disable.
_C.VIS_PERIOD = 0

# global config is for quick hack purposes.
# You can set them in command line or config files,
# and access it with:
#
# from detectron2.config import global_cfg
# print(global_cfg.HACK)
#
# Do not commit any configs into it.
_C.GLOBAL = CN()
_C.GLOBAL.HACK = 1.0
