import argparse


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='RGMP')
  parser.add_argument('--epochs', dest='num_epochs',
                      help='number of epochs to train',
                      default=400, type=int)
  parser.add_argument('--bs', dest='bs',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--num_workers', dest='num_workers',
                      help='num_workers',
                      default=1, type=int)
  parser.add_argument('--disp_interval', dest='disp_interval',
                      help='display interval',
                      default=10, type=int)
  parser.add_argument('--eval_epoch', dest='eval_epoch',
                      help='interval of epochs to perform validation',
                      default=10, type=int)
  parser.add_argument('--results_path', dest='results_path',
                      help='path to save results',
                      default=None, type=str)
  parser.add_argument('--save_results', dest='save_results',
                      help='save predictions during inference',
                      default=False, type=bool)
  parser.add_argument('--proposal_dir', dest='proposal_dir',
                      help='path to proposals',
                      default="results/converted_proposals/thresh-0-all_fields/", type=str)
  parser.add_argument('--remove_gt_proposal', dest='remove_gt_proposal',
                      help='remove the gt proposal randomly',
                      default=False, type=bool)
  parser.add_argument('--freeze_bn', dest='freeze_bn',
                      help='freeze batch normalisation layers',
                      default=False, type=bool)
  parser.add_argument('--crop_size', dest='crop_size',
                      help='image crop size', nargs = '+',
                      default=None, type=int)
  parser.add_argument('--resize_mode', dest='resize_mode',
                      help='resize mode',
                      default="fixed_size", type=str)
  parser.add_argument('--crop_size_eval', dest='crop_size_eval',
                      help='image crop size', nargs = '+',
                      default=None, type=int)
  parser.add_argument('--resize_mode_eval', dest='resize_mode_eval',
                      help='resize mode',
                      default="unchanged", type=str)
  parser.add_argument('--data_sample', dest='data_sample',
                      help='number of data samples',
                      default=None, type=int)
  parser.add_argument('--random_instance', dest='random_instance',
                      help='random_instance',
                      default=False, type=bool)
  parser.add_argument('--train_dataset', dest='train_dataset',
                      help='train dataset',
                      default="davis", type=str)
  parser.add_argument('--test_dataset', dest='test_dataset',
                      help='test dataset',
                      default="davis", type=str)
  parser.add_argument('--tw', dest='tw',
                      help='temporal window size',
                      default=5, type=int)
  parser.add_argument('--max_temporal_gap', dest='max_temporal_gap',
                      help='maximum temporal gap relative to current frame from which the input clip should be sampled',
                      default=5, type=int)
  parser.add_argument('--augmentors', dest='augmentors',
                      help='augmentors to use',
                      nargs='*',
                      default=None, type=str)

  # BPTT
  parser.add_argument('--bptt', dest='bptt_len',
                      help='length of BPTT',
                      default=12, type=int)
  parser.add_argument('--bptt_step', dest='bptt_step',
                      help='step of truncated BPTT',
                      default=4, type=int)

  # config optimization
  parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="sgd", type=str)
  parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=1e-3, type=float)
  parser.add_argument('--adjust_lr', dest='adjust_lr',
                      help='use adaptive learning rate',
                      default=1e-3, type=float)
  parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=5, type=int)
  parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)
  parser.add_argument('--loss_scale', dest='loss_scale',
                      help='scale factor for scaling the loss value',
                      default=1, type=float)

  # resume trained model
  parser.add_argument('--loadepoch', dest='loadepoch',
                      help='epoch to load model',
                      default=None, type=str)

  parser.add_argument('--network', dest='network',
                      help='Network to use',
                      default=0, type=int)
  parser.add_argument('--MO', dest='MO',
                      help='Network to use',
                      default=False, type=bool)
  parser.add_argument('--network_name', dest='network_name',
                      help='Network name',
                      default=None, type=str, required=True)
  parser.add_argument('--task', dest='task',
                      help='task in <train, eval>',
                      default='train', type=str)
  parser.add_argument('--update_refs', dest='update_refs',
                      help='Update reference objects during evaluation',
                      default=False, type=bool)
  parser.add_argument('--lr_schedulers', dest='lr_schedulers',
                      help='specify a list of learning rate schedulers',
                      nargs='*',
                      default=None, type=str)
  parser.add_argument('--lr_decay', dest='lr_decay',
                      help='learning rate decay rate',
                      default=0.95, type=float)

  # tensorboard summary
  parser.add_argument('--show_image_summary', dest='show_image_summary',
                      help='show image summary',
                      default=False, type=bool)

  args = parser.parse_args()
  return args
