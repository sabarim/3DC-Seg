import torch
from torch import nn
from torch.nn import functional as F

from network.Modules import GC
from network.RGMP import Encoder, Decoder
from network.models import BaseNetwork


class Decoder3d(Decoder):
  def __init__(self):
    super(Decoder3d, self).__init__()
    self.conv3d_1 = nn.Sequential(nn.Conv3d(in_channels=2048, out_channels=1024, kernel_size=(3,3,3), padding=1),
                                  nn.LeakyReLU())
    self.conv3d_2 = nn.Sequential(nn.Conv3d(in_channels=1024, out_channels=512, kernel_size=(3, 3, 3), padding=1),
                                  nn.LeakyReLU())
    self.GC = GC(512, 256)

  def forward(self, r5, r4, r3, r2, support):
    x = torch.cat((r5.unsqueeze(2), support), dim=2)
    x = self.conv3d_1(x)
    x = self.conv3d_2(x)[:, :, -1]

    x = self.GC(x)
    r = self.convG1(F.relu(x))
    r = self.convG2(F.relu(r))
    m5 = x + r  # out: 1/32, 64
    m4 = self.RF4(r4, m5)  # out: 1/16, 64
    m3 = self.RF3(r3, m4)  # out: 1/8, 64
    m2 = self.RF2(r2, m3)  # out: 1/4, 64

    p2 = self.pred2(F.relu(m2))
    p3 = self.pred3(F.relu(m3))
    p4 = self.pred4(F.relu(m4))
    p5 = self.pred5(F.relu(m5))

    p = F.upsample(p2, scale_factor=4, mode='bilinear')

    return p, p2, p3, p4, p5


class FeatureAgg3d(BaseNetwork):
  def __init__(self):
    super(FeatureAgg3d, self).__init__()
    self.encoder = Encoder()
    self.decoder = Decoder3d()
