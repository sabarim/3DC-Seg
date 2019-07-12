import torch
from torch import nn
from torch.nn import functional as F

from network.Modules import GC
from network.RGMP import Encoder, Decoder
from network.models import BaseNetwork


class Decoder3d(Decoder):
  def __init__(self):
    super(Decoder3d, self).__init__()
    self.GC = GC(512, 256)
    self.temporal_net = TemporalNetSmall()

  def forward(self, r5, r4, r3, r2, support):
    x = self.temporal_net(r5, support)
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

    p = F.interpolate(p2, scale_factor=4, mode='bilinear')

    return p, p2, p3, p4, p5


class Decoder3dMergeTemporal(Decoder3d):
  def __init__(self, tw=5 ):
    super(Decoder3dMergeTemporal, self).__init__()
    self.temporal_net = TemporalNet(tw)
    self.GC = GC(256,256)


class TemporalNet(BaseNetwork):
  def __init__(self,  tw=5):
    super(TemporalNet, self).__init__()
    self.conv3d_1 = nn.Sequential(nn.Conv3d(in_channels=2048, out_channels=1024, kernel_size=(3, 3, 3), padding=1),
                                  nn.BatchNorm3d(1024), nn.LeakyReLU())
    self.conv3d_2 = nn.Sequential(nn.Conv3d(in_channels=1024, out_channels=512, kernel_size=(3, 3, 3), padding=1),
                                  nn.BatchNorm3d(512), nn.LeakyReLU())
    self.conv3d_3 = nn.Sequential(nn.Conv3d(in_channels=512, out_channels=256, kernel_size=(3, 3, 3), padding=1),
                                  nn.BatchNorm3d(256), nn.LeakyReLU())
    self.conv3d_4 = nn.Sequential(nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(3, 1, 1), padding=0),
                                  nn.BatchNorm3d(256), nn.LeakyReLU())
    self.conv3d_5 = nn.Sequential(nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(3, 1, 1), padding=0),
                                  nn.BatchNorm3d(256), nn.LeakyReLU())

  def forward(self, r5, support):
    x = torch.cat((r5.unsqueeze(2), support), dim=2)
    x = self.conv3d_1(x)
    x = self.conv3d_2(x)
    x = self.conv3d_3(x)
    x = self.conv3d_4(x)
    x = self.conv3d_5(x)

    return x[:, :, -1]


class TemporalNetSmall(BaseNetwork):
  def __init__(self, tw=5):
    super(TemporalNetSmall, self).__init__()
    self.conv3d_1 = nn.Sequential(nn.Conv3d(in_channels=2048, out_channels=1024, kernel_size=(3, 3, 3), padding=1),
                                  nn.LeakyReLU())
    self.conv3d_2 = nn.Sequential(nn.Conv3d(in_channels=1024, out_channels=512, kernel_size=(3, 3, 3), padding=1),
                                  nn.LeakyReLU())

  def forward(self, r5, support):
    x = torch.cat((r5.unsqueeze(2), support), dim=2)
    x = self.conv3d_1(x)
    x = self.conv3d_2(x)[:, :, -1]
    return x


class FeatureAgg3d(BaseNetwork):
  def __init__(self):
    super(FeatureAgg3d, self).__init__()
    self.encoder = Encoder()
    self.decoder = Decoder3d()


class FeatureAgg3dMergeTemporal(BaseNetwork):
  def __init__(self):
    super(FeatureAgg3dMergeTemporal, self).__init__()
    self.encoder = Encoder()
    self.decoder = Decoder3dMergeTemporal()
