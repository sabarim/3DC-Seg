import torch
from torch import nn
from torchvision import models
from torch.nn import functional as F
from network.Modules import Refine, GC
from network.models import BaseNetwork


class Encoder(nn.Module):
  def __init__(self, n_classes=1):
    super(Encoder, self).__init__()

    self.conv1_p = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=True)

    resnet = models.resnet50(pretrained=True)
    self.conv1 = resnet.conv1
    self.bn1 = resnet.bn1
    self.relu = resnet.relu  # 1/2, 64
    self.maxpool = resnet.maxpool

    self.res2 = resnet.layer1  # 1/4, 256
    self.res3 = resnet.layer2  # 1/8, 512
    self.res4 = resnet.layer3  # 1/16, 1024
    self.res5 = resnet.layer4  # 1/32, 2048

    self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
    self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

  def freeze_batchnorm(self):
    # freeze BNs
    print("Freezing batchnorm for Resnet50")
    for m in self.modules():
      if isinstance(m, nn.BatchNorm2d):
        for p in m.parameters():
          p.requires_grad = False

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
    r4 = self.res4(r3)  # 1/16, 256
    r5 = self.res5(r4)  # 1/32, 512

    return r5, r4, r3, r2


class Decoder(nn.Module):
  def __init__(self):
    super(Decoder, self).__init__()
    mdim = 256
    self.GC = GC(4096, mdim)  # 1/32 -> 1/32
    self.convG1 = nn.Conv2d(mdim, mdim, kernel_size=3, padding=1)
    self.convG2 = nn.Conv2d(mdim, mdim, kernel_size=3, padding=1)
    self.RF4 = Refine(1024, mdim)  # 1/16 -> 1/8
    self.RF3 = Refine(512, mdim)  # 1/8 -> 1/4
    self.RF2 = Refine(256, mdim)  # 1/4 -> 1

    self.pred5 = nn.Conv2d(mdim, 2, kernel_size=(3, 3), padding=(1, 1), stride=1)
    self.pred4 = nn.Conv2d(mdim, 2, kernel_size=(3, 3), padding=(1, 1), stride=1)
    self.pred3 = nn.Conv2d(mdim, 2, kernel_size=(3, 3), padding=(1, 1), stride=1)
    self.pred2 = nn.Conv2d(mdim, 2, kernel_size=(3, 3), padding=(1, 1), stride=1)

  def forward(self, r5, x5, r4, r3, r2):
    x = torch.cat((r5, x5), dim=1)

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


class RGMP(BaseNetwork):
  def __init__(self):
    super(RGMP, self).__init__()
    self.encoder = Encoder()
    self.decoder = Decoder()
