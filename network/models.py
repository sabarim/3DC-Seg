from torch import nn


class BaseNetwork(nn.Module):
  def __init__(self):
    super(BaseNetwork, self).__init__()