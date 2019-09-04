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


class Refine3d(nn.Module):
  def __init__(self, inplanes, planes, scale_factor=2):
    super(Refine3d, self).__init__()
    self.convFS1 = nn.Conv3d(inplanes, planes, kernel_size=3, padding=1)
    self.convFS2 = nn.Conv3d(planes, planes, kernel_size=3, padding=1)
    self.convFS3 = nn.Conv3d(planes, planes, kernel_size=3, padding=1)
    self.convMM1 = nn.Conv3d(planes, planes, kernel_size=3, padding=1)
    self.convMM2 = nn.Conv3d(planes, planes, kernel_size=3, padding=1)
    self.scale_factor = scale_factor

  def forward(self, f, pm):
    s = self.convFS1(f)
    sr = self.convFS2(F.relu(s))
    sr = self.convFS3(F.relu(sr))
    s = s + sr

    m = s + F.upsample(pm, scale_factor=self.scale_factor, mode='trilinear')

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


class GC3d(nn.Module):
  def __init__(self, inplanes, planes, kh=7, kw=7):
    super(GC3d, self).__init__()
    self.conv_l1 = nn.Conv3d(inplanes, 256, kernel_size=(1,kh, 1),
                             padding=(0, int(kh / 2), 0))
    self.conv_l2 = nn.Conv3d(256, planes, kernel_size=(1, 1, kw),
                             padding=(0, 0, int(kw / 2)))
    self.conv_r1 = nn.Conv3d(inplanes, 256, kernel_size=(1, 1, kw),
                             padding=(0, 0, int(kw / 2)))
    self.conv_r2 = nn.Conv3d(256, planes, kernel_size=(1, kh, 1),
                             padding=(0, int(kh / 2), 0))

  def forward(self, x):
    x_l = self.conv_l2(self.conv_l1(x))
    x_r = self.conv_r2(self.conv_r1(x))
    x = x_l + x_r
    return x


class SoftmaxSimilarity(nn.Module):
  def __init__(self, apply_softmax):
    super(SoftmaxSimilarity, self).__init__()
    self.apply_softmax = apply_softmax
    self.softmax = torch.nn.functional.softmax

  def forward(self, x):
    # Check if it is a spacetime input
    num_features = x.shape[1]
    m1 = x[:, :int(num_features/2)]
    m2 = x[:, int(num_features/2):]
    shape = m1.shape
    m1 = m1.permute(0, 2, 3, 1).view(m1.shape[0], m1.shape[2] * m1.shape[3], m1.shape[1])
    m2 = m2.permute(0, 2, 3, 1).view(m2.shape[0], m2.shape[2] * m2.shape[3], m2.shape[1]).transpose(1, 2)

    output = torch.bmm(m1, m2)
    # output = output.view(shape[0], shape[2] * shape[3], shape[2], shape[3])
    if self.apply_softmax:
        output = self.softmax(output, dim=1)
    # output = output / output.sum(dim=3)[..., None]
    # x[DataKeys.INPUTS] = output.reshape(shape[0], shape[2], shape[3], shape[2]*shape[3]).transpose(1,3)
    x = output.transpose(1, 2).view(shape[0], shape[2] * shape[3], shape[2], shape[3])
    del m1,m2
    return x