from functools import reduce
from network.NonLocal import NONLocalBlock3D

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.segmentation import deeplabv3_resnet101, fcn_resnet101

from network.Modules import GC3d, Refine3d
from network.R2plus1d import r2plus1d_34
from network.RGMP import Encoder
from network.Resnet3d import resnet50, resnet152_csn_ip, resnet152_csn_ir, resnet101
from network.models import BaseNetwork


class Encoder3d(Encoder):
  def __init__(self, tw = 16, sample_size = 112, resnet = None):
    super(Encoder3d, self).__init__()
    self.conv1_p = nn.Conv3d(1, 64, kernel_size=7, stride=(1, 2, 2),
                             padding=(3, 3, 3), bias=False)

    resnet = resnet50(sample_size = sample_size, sample_duration = tw) if resnet is None else resnet
    self.resnet = resnet
    self.conv1 = resnet.conv1
    self.bn1 = resnet.bn1
    self.relu = resnet.relu  # 1/2, 64
    # self.maxpool = resnet.maxpool
    self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1,2,2), padding=(0,1,1))

    self.layer1 = resnet.layer1  # 1/4, 256
    self.layer2 = resnet.layer2  # 1/8, 512
    self.layer3 = resnet.layer3  # 1/16, 1024
    self.layer4 = resnet.layer4  # 1/32, 2048

    self.register_buffer('mean', torch.FloatTensor([114.7748, 107.7354, 99.4750]).view(1, 3, 1, 1, 1))
    self.register_buffer('std', torch.FloatTensor([1, 1, 1]).view(1, 3, 1, 1, 1))

  def freeze_batchnorm(self):
    print("Freezing batchnorm for Encoder3d")
    # freeze BNs
    for m in self.modules():
      if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
        for p in m.parameters():
          p.requires_grad = False

  def forward(self, in_f, in_p=None):
    assert in_f is not None or in_p is not None
    f = (in_f * 255.0 - self.mean) / self.std
    f /= 255.0

    if in_f is None:
      p = in_p
      if len(in_p.shape) < 4:
        p = torch.unsqueeze(in_p, dim=1)  # add channel dim

      x = self.conv1_p(p)
    elif in_p is not None:
      p = in_p
      if len(in_p.shape) < 4:
        p = torch.unsqueeze(in_p, dim=1)  # add channel dim

      x = self.conv1(f) + self.conv1_p(p)  # + self.conv1_n(n)
    else:
      x = self.conv1(f)
    x = self.bn1(x)
    c1 = self.relu(x)  # 1/2, 64
    x = self.maxpool(c1)  # 1/4, 64
    r2 = self.layer1(x)  # 1/4, 64
    r3 = self.layer2(r2)  # 1/8, 128
    r4 = self.layer3(r3)  # 1/16, 256
    r5 = self.layer4(r4)  # 1/32, 512

    return r5, r4, r3, r2


class Encoder101(Encoder):
  def __init__(self):
    super(Encoder101, self).__init__()

    self.resnet = deeplabv3_resnet101(pretrained=True)
    self.conv1_p = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=True)

    resnet = fcn_resnet101(pretrained=True)
    self.conv1 = resnet.backbone.conv1
    self.bn1 = resnet.backbone.bn1
    self.relu = resnet.backbone.relu  # 1/2, 64
    self.maxpool = resnet.backbone.maxpool

    self.res2 = resnet.backbone.layer1  # 1/4, 256
    self.res3 = resnet.backbone.layer2  # 1/8, 512
    self.res4 = resnet.backbone.layer3  # 1/16, 1024
    self.res5 = resnet.backbone.layer4  # 1/32, 2048

    self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
    self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))


class EncoderR2plus1d_34(Encoder3d):
  def __init__(self, tw = 8, sample_size = 112):
    super(EncoderR2plus1d_34, self).__init__(tw, sample_size)
    resnet = r2plus1d_34(num_classes=359, pretrained=True, arch='r2plus1d_34_32_ig65m')
    self.resnet = resnet
    self.conv1 = resnet.stem
    self.bn1 = nn.Identity()
    self.relu = nn.Identity()

    self.layer1 = resnet.layer1  
    self.layer2 = resnet.layer2  
    self.layer3 = resnet.layer3  
    self.layer4 = resnet.layer4  

    
class Encoder3d_csn_ip(Encoder3d):
  def __init__(self, tw = 16, sample_size = 112):
    super(Encoder3d_csn_ip, self).__init__(tw, sample_size)
    resnet = resnet152_csn_ip(sample_size = sample_size, sample_duration = tw)
    self.resnet = resnet
    self.conv1 = resnet.conv1
    self.bn1 = resnet.bn1
    self.relu = resnet.relu  # 1/2, 64

    self.layer1 = resnet.layer1  # 1/4, 256
    self.layer2 = resnet.layer2  # 1/8, 512
    self.layer3 = resnet.layer3  # 1/16, 1024
    self.layer4 = resnet.layer4  # 1/32, 2048


class Encoder3d_csn_ir(Encoder3d):
  def __init__(self, tw = 16, sample_size = 112):
    super(Encoder3d_csn_ir, self).__init__(tw, sample_size)
    resnet = resnet152_csn_ir(sample_size = sample_size, sample_duration = tw)
    self.resnet = resnet
    self.conv1 = resnet.conv1
    self.bn1 = resnet.bn1
    self.relu = resnet.relu  # 1/2, 64

    self.layer1 = resnet.layer1  # 1/4, 256
    self.layer2 = resnet.layer2  # 1/8, 512
    self.layer3 = resnet.layer3  # 1/16, 1024
    self.layer4 = resnet.layer4  # 1/32, 2048


class Decoder3d(nn.Module):
  def __init__(self, n_classes=2, pred_scale_factor = (1,4,4), inter_block=GC3d, refine_block = Refine3d):
    super(Decoder3d, self).__init__()
    mdim = 256
    self.pred_scale_factor = pred_scale_factor
    self.GC = inter_block(2048, mdim)
    self.convG1 = nn.Conv3d(mdim, mdim, kernel_size=3, padding=1)
    self.convG2 = nn.Conv3d(mdim, mdim, kernel_size=3, padding=1)
    self.RF4 = refine_block(1024, mdim)  # 1/16 -> 1/8
    self.RF3 = refine_block(512, mdim)  # 1/8 -> 1/4
    self.RF2 = refine_block(256, mdim)  # 1/4 -> 1

    self.pred5 = nn.Conv3d(mdim, n_classes, kernel_size=3, padding=1, stride=1)
    self.pred4 = nn.Conv3d(mdim, n_classes, kernel_size=3, padding=1, stride=1)
    self.pred3 = nn.Conv3d(mdim, n_classes, kernel_size=3, padding=1, stride=1)
    self.pred2 = nn.Conv3d(mdim, n_classes, kernel_size=3, padding=1, stride=1)

  def forward(self, r5, r4, r3, r2, support):
    # there is a merge step in the temporal net. This split is a hack to fool it
    # x = torch.cat((x, r5), dim=1)
    x = self.GC(r5)
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

    p = F.interpolate(p2, scale_factor=self.pred_scale_factor, mode='trilinear')

    return p


class Decoder3dNoGC(Decoder3d):
  def __init__(self, n_classes=2):
    super(Decoder3dNoGC, self).__init__(n_classes=n_classes)
    self.GC = nn.Conv3d(2048, 256, kernel_size=3, padding=1)


class Decoder3dNonLocal(Decoder3d):
  def __init__(self, n_classes=2):
    super(Decoder3dNonLocal, self).__init__(n_classes=n_classes)
    self.GC = nn.Sequential(nn.Conv3d(2048, 256, kernel_size=1),
                            NONLocalBlock3D(256, sub_sample=True))


class DecoderR2plus1d(Decoder3d):
  def __init__(self, n_classes=2, inter_block=GC3d, refine_block = Refine3d):
    super(DecoderR2plus1d, self).__init__(n_classes=n_classes)
    mdim=256
    self.GC = inter_block(512, 256)
    self.RF4 = refine_block(256, mdim)  # 1/16 -> 1/8
    self.RF3 = refine_block(128, mdim)  # 1/8 -> 1/4
    self.RF2 = refine_block(64, mdim)


class Resnet3d(BaseNetwork):
  def __init__(self, tw=16, sample_size=112):
    super(Resnet3d, self).__init__()
    self.encoder = Encoder3d(tw, sample_size)
    self.decoder = Decoder3d()

  def forward(self, x, ref):
    if ref is not None and len(ref.shape) == 4:
      r5, r4, r3, r2 = self.encoder.forward(x, ref.unsqueeze(2))
    else:
      r5, r4, r3, r2 = self.encoder.forward(x, ref)
    p = self.decoder.forward(r5, r4, r3, r2, None)
    return [p]


class Resnet3d101(Resnet3d):
  def __init__(self, tw=8, sample_size=112, e_dim=7, decoders=None, inter_block=GC3d, refine_block = Refine3d):
    super(Resnet3d101, self).__init__(tw=tw, sample_size=sample_size)
    resnet = resnet101(sample_size=sample_size, sample_duration=tw)
    self.encoder = Encoder3d(tw, sample_size, resnet=resnet)
    decoders = [Decoder3d(inter_block=inter_block, refine_block = refine_block)] if decoders is None else decoders
    self.decoders = nn.ModuleList()
    for decoder in decoders:
      self.decoders.append(decoder)
    print("Using decoders {}".format(self.decoders))

  def forward(self, x, ref = None):
    r5, r4, r3, r2 = self.encoder.forward(x, ref)
    flatten = lambda lst: [lst] if type(lst) is torch.Tensor else reduce(torch.add, [flatten(ele) for ele in lst])
    p = flatten([decoder.forward(r5, r4, r3, r2, None) for decoder in self.decoders])
    # e = self.decoder_embedding.forward(r5, r4, r3, r2, None)
    return p


class R2plus1d(Resnet3d101):
  def __init__(self, tw=8, sample_size=112, e_dim=7, decoders=None, inter_block=GC3d, refine_block = Refine3d):
    decoders = [DecoderR2plus1d(inter_block=inter_block, refine_block=refine_block)]
    super(R2plus1d, self).__init__(tw, sample_size, e_dim, decoders)
    self.encoder = EncoderR2plus1d_34(tw, sample_size)


class ResnetCSN(Resnet3d101):
  def __init__(self, tw=8, sample_size=112, e_dim=7, decoders=None, inter_block=GC3d, refine_block = Refine3d):
    super(ResnetCSN, self).__init__(tw, sample_size, e_dim, decoders, inter_block=inter_block, refine_block=refine_block)
    self.encoder = Encoder3d_csn_ir(tw, sample_size)


class ResnetCSNNoGC(Resnet3d101):
  def __init__(self, tw=8, sample_size=112, e_dim=7, decoders=None):
    decoders = [Decoder3dNoGC()] if decoders is None else decoders
    print("Creating decoders {}".format(decoders))
    super(ResnetCSNNoGC, self).__init__(tw, sample_size, e_dim, decoders)
    self.encoder = Encoder3d_csn_ir(tw, sample_size)


class ResnetCSNNonLocal(ResnetCSNNoGC):
  def __init__(self, tw=8, sample_size=112, e_dim=7):
    decoders = [Decoder3dNonLocal()]
    super(ResnetCSNNonLocal, self).__init__(tw, sample_size, e_dim, decoders)
