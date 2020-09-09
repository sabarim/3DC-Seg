import argparse


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_argsV2():
  parser = argparse.ArgumentParser(description='SaliencySegmentation')
  parser.add_argument('--config', "-c", required=True, type=str)
  parser.add_argument('--num_workers', dest='num_workers',
                      help='num_workers',
                      default=4, type=int)
  parser.add_argument('--local_rank', type=int, default=0)
  parser.add_argument('--print_freq', dest='print_freq',
                      help='Frequency of statistics printing',
                      default=1, type=int)
  # resume trained model
  parser.add_argument('--loadepoch', dest='loadepoch',
                      help='epoch to load model',
                      default=None, type=str)

  parser.add_argument('--task', dest='task',
                      help='task in <train, eval>',
                      default='train', type=str)
  parser.add_argument('--pretrained', dest='pretrained',
                      help='load pretrained weights for PWCNet',
                      default='weights/pwc_net.pth.tar', type=str)
  parser.add_argument('--wts', '-w', dest='wts',
                      help='weights file to resume training',
                      default=None, type=str)

  # summary generation
  parser.add_argument('--show_image_summary', dest='show_image_summary',
                      help='load the best model',
                      default=False, type=bool)
  args = parser.parse_args()
  return args