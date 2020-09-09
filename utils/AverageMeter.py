import pprint


class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


class AverageMeterDict(object):
  """Computes and stores the average and current value"""

  def __init__(self):
    self.reset()

  def reset(self):
    self.val = {}
    self.avg = {}
    self.sum = None
    self.count = 0

  def update(self, in_dict, n=1):
    self.val = in_dict
    self.sum = in_dict if self.sum is None else dict([(key, val * n + self.sum[key]) for key, val in in_dict.items()])
    self.count += n
    self.avg = dict([(key, (val / self.count)) for key, val in self.sum.items()])

  def __str__(self):
    val = dict([(key,"{:.4f}".format(val)) for key, val in self.val.items()])
    avg = dict([(key,"{:.4f}".format(val)) for key, val in self.avg.items()])
    return str(val) + '(' + str(avg) + ')'