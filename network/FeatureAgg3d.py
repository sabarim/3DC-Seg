import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

from network.Modules import GC, SoftmaxSimilarity
from network.RGMP import Encoder, Decoder
from network.models import BaseNetwork


class EncoderWG(Encoder):
  def __init__(self, tw=5):
    super(EncoderWG, self).__init__()
    self.conv1_p = nn.Conv2d(tw, 64, kernel_size=7, stride=2, padding=3, bias=True)

  def forward(self, in_f, in_p):
    super(EncoderWG, self).forward(in_f, in_p)


class Encoder101(Encoder):
  def __init__(self):
    super(Encoder101, self).__init__()
    resnet = models.segmentation.fcn_resnet101(pretrained=True)
    self.resnet = resnet
    self.conv1 = resnet.backbone.conv1
    self.bn1 = resnet.backbone.bn1
    self.relu = resnet.backbone.relu  # 1/2, 64
    self.maxpool = resnet.backbone.maxpool

    self.res2 = resnet.backbone.layer1  # 1/4, 256
    self.res3 = resnet.backbone.layer2  # 1/8, 512
    self.res4 = resnet.backbone.layer3  # 1/16, 1024
    self.res5 = resnet.backbone.layer4  # 1/32, 2048

  def forward(self, in_f, in_p):
    assert in_f is not None or in_p is not None
    f = (in_f - self.mean) / self.std

    if in_f is None:
      p = in_p.float()
      if len(in_p.shape) < 4:
        p = torch.unsqueeze(in_p, dim=1).float()  # add channel dim

      x = self.conv1_p(p)
    elif in_p is not None:
      p = in_p.float()
      if len(in_p.shape) < 4:
        p = torch.unsqueeze(in_p, dim=1).float()  # add channel dim

      x = self.conv1(f) + self.conv1_p(p)  # + self.conv1_n(n)
    else:
      x = self.conv1(f)
    x = self.bn1(x)
    c1 = self.relu(x)  # 1/2, 64
    x = self.maxpool(c1)  # 1/4, 64
    r2 = self.res2(x)  # 1/4, 64
    r3 = self.res3(r2)  # 1/8, 128
    r4 = self.res4(r3)
    r4 = self.maxpool(r4)# 1/16, 256
    r5 = self.res5(r4)
    r5 = self.maxpool(r5)# 1/32, 512

    return r5, r4, r3, r2


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


class DecoderSM(Decoder3d):
  def __init__(self, tw=5):
    super(DecoderSM, self).__init__()
    self.temporal_net = TemporalNetNoMerge()
    self.convF1 = nn.Conv2d(in_channels=(tw-1)*256, out_channels=2048, kernel_size=1)
    self.GC = GC(4096, 256)

  def forward(self, r5, r4, r3, r2, support):
    # there is a merge step in the temporal net. This split is a hack to fool it
    x = self.temporal_net(support[:, :, -1], support[:, :, :-1])
    x = x.reshape(tuple(x.shape[:1]) + (-1,) + tuple(x.shape[3:],))
    x = self.convF1(F.relu(x))
    x = torch.cat((x, r5), dim=1)
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


class DecoderPredictTemporal(DecoderSM):
  def __init__(self, tw=5):
    super(DecoderPredictTemporal, self).__init__(tw)


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

  def forward(self, r5, support, no_merge=False):
    x = torch.cat((r5.unsqueeze(2), support), dim=2)
    x = self.conv3d_1(x)
    x = self.conv3d_2(x)
    x = self.conv3d_3(x)
    x = self.conv3d_4(x)
    x = self.conv3d_5(x)

    if no_merge:
      return x
    else:
      return x[:, :, -1]


class TemporalNetNoMerge(TemporalNet):
  def __init__(self):
    super(TemporalNetNoMerge, self).__init__()
    self.conv3d_4 = nn.Sequential(nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(3, 3, 3), padding=1),
                                  nn.BatchNorm3d(256), nn.LeakyReLU())
    self.conv3d_5 = nn.Sequential(nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(3, 3, 3), padding=1),
                                  nn.BatchNorm3d(256), nn.LeakyReLU())

  def forward(self, r5, support, no_merge=True):
    return super(TemporalNetNoMerge, self).forward(r5, support, True)


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


class TemporalAssociation(TemporalNet):
  def __init__(self, tw=5):
    super(TemporalAssociation, self).__init__()
    self.conv3d_4 = nn.Sequential(nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(3, 1, 1), padding=1),
                                  nn.BatchNorm3d(256), nn.LeakyReLU())
    self.conv3d_5 = nn.Sequential(nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(3, 1, 1), padding=1),
                                  nn.BatchNorm3d(256), nn.LeakyReLU())
    self.conv1x1 = nn.Sequential(nn.Conv2d(in_channels= 256*(tw-1), out_channels=256, kernel_size=1))
    self.similarity = SoftmaxSimilarity(apply_softmax=True)

  def forward(self, r5, support):
    x = torch.cat((r5.unsqueeze(2), support), dim=2)
    x = self.conv3d_1(x)
    x = self.conv3d_2(x)
    x = self.conv3d_3(x)
    x = self.conv3d_4(x)
    x = self.conv3d_5(x)

    x1 = torch.cat(x[:, :-1], dim=1)
    x1 = self.conv1x1(x1)
    x_sim = self.similarity(torch.cat((x1, x[:, :, -1]), dim=1))

    return x[:, :, -1], x_sim


class FeatureAgg3d(BaseNetwork):
  def __init__(self):
    super(FeatureAgg3d, self).__init__()
    self.encoder = Encoder()
    self.decoder = Decoder3d()


class FeatureAgg3dMergeTemporal(BaseNetwork):
  def __init__(self, tw=5):
    super(FeatureAgg3dMergeTemporal, self).__init__()
    self.encoder = Encoder()
    self.decoder = Decoder3dMergeTemporal()


class FeatureAgg3dTemporalAssociation(BaseNetwork):
  def __init__(self):
    super(FeatureAgg3dTemporalAssociation, self).__init__()
    self.encoder = Encoder()
    self.decoder = Decoder3dMergeTemporal()


class FeatureAgg3dMulti(BaseNetwork):
  def __init__(self, tw=5):
    super(FeatureAgg3dMulti, self).__init__(tw=tw)
    self.encoder = Encoder()
    self.encoder2 = Encoder()
    self.decoder = DecoderSM(tw=tw)


class FeatureAgg3dMulti101(BaseNetwork):
  def __init__(self, tw=5):
    super(FeatureAgg3dMulti101, self).__init__(tw=tw)
    self.encoder = Encoder101()
    self.encoder2 = Encoder101()
    self.decoder = DecoderSM(tw=tw)
