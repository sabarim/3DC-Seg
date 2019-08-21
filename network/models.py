from torch import nn


class BaseNetwork(nn.Module):
  def __init__(self, tw=5):
    super(BaseNetwork, self).__init__()
    self.tw = tw