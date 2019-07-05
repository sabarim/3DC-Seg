from torch import nn
from torch.nn import functional as F

class Refine(nn.Module):
  def __init__(self, inplanes, planes, scale_factor=2):
    super(Refine, self).__init__()
    self.convFS1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1)
    self.convFS2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
    self.convFS3 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
    self.convMM1 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
    self.convMM2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
    self.scale_factor = scale_factor

  def forward(self, f, pm):
    s = self.convFS1(f)
    sr = self.convFS2(F.relu(s))
    sr = self.convFS3(F.relu(sr))
    s = s + sr

    m = s + F.upsample(pm, scale_factor=self.scale_factor, mode='bilinear')

    mr = self.convMM1(F.relu(m))
    mr = self.convMM2(F.relu(mr))
    m = m + mr
    return m

class GC(nn.Module):
  def __init__(self, inplanes, planes, kh=7, kw=7):
    super(GC, self).__init__()
    self.conv_l1 = nn.Conv2d(inplanes, 256, kernel_size=(kh, 1),
                             padding=(int(kh / 2), 0))
    self.conv_l2 = nn.Conv2d(256, planes, kernel_size=(1, kw),
                             padding=(0, int(kw / 2)))
    self.conv_r1 = nn.Conv2d(inplanes, 256, kernel_size=(1, kw),
                             padding=(0, int(kw / 2)))
    self.conv_r2 = nn.Conv2d(256, planes, kernel_size=(kh, 1),
                             padding=(int(kh / 2), 0))

  def forward(self, x):
    x_l = self.conv_l2(self.conv_l1(x))
    x_r = self.conv_r2(self.conv_r1(x))
    x = x_l + x_r
    return x