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
  parser.add_argument('--remove_gt_proposal', dest='remove_gt_proposal',
                      help='remove the gt proposal randomly',
                      default=False, type=bool)
  parser.add_argument('--freeze_bn', dest='freeze_bn',
                      help='freeze batch normalisation layers',
                      default=False, type=bool)
  parser.add_argument('--crop_size', dest='crop_size',
                      help='image crop size',
                      default=None, type=int)
  parser.add_argument('--data_sample', dest='data_sample',
                      help='number of data samples',
                      default=None, type=int)

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
                      default=None, type=str)
  parser.add_argument('--task', dest='task',
                      help='task in <train, eval>',
                      default='train', type=str)
  parser.add_argument('--update_refs', dest='update_refs',
                      help='Update reference objects during evaluation',
                      default=False, type=bool)
  parser.add_argument('--adaptive_lr', dest='adaptive_lr',
                      help='use an lr scheduler',
                      default=False, type=bool)

  # tensorboard summary
  parser.add_argument('--show_image_summary', dest='show_image_summary',
                      help='show image summary',
                      default=False, type=bool)

  args = parser.parse_args()
  return args