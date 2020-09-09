import torch
import math
from torch import nn
from torch.nn import functional as F
from torchvision.models.video.resnet import Conv2Plus1D

from network.NonLocal import NONLocalBlock3D


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

    m = s + F.interpolate(pm, scale_factor=self.scale_factor, mode='bilinear')

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

    m = s + F.interpolate(pm, size=s.shape[-3:], mode='trilinear')

    mr = self.convMM1(F.relu(m))
    mr = self.convMM2(F.relu(mr))
    m = m + mr
    return m


class RefineSimple(nn.Module):
  def __init__(self, inplanes, planes, scale_factor=2):
    super(RefineSimple, self).__init__()
    self.convMM1 = nn.Conv3d(planes + inplanes, planes, kernel_size=3, padding=1)
    self.convMM2 = nn.Conv3d(planes, planes, kernel_size=3, padding=1)
    self.scale_factor = scale_factor

  def forward(self, f, pm):
    m = torch.cat((f, F.interpolate(pm, size=f.shape[-3:], mode='trilinear')), dim=1)

    m = self.convMM1(m)
    m = self.convMM2(F.relu(m))
    return m


class Refine2plus1d(Refine3d):
  def __init__(self, inplanes, planes, scale_factor=2):
    super(Refine2plus1d, self).__init__(inplanes, planes, scale_factor)
    self.convFS1 = Conv2Plus1D(inplanes, planes, planes*2 + 32)
    self.convFS2 = Conv2Plus1D(planes, planes, planes*2 + 32)
    self.convFS3 = Conv2Plus1D(planes, planes, planes*2 + 32)
    self.convMM1 = Conv2Plus1D(planes, planes, planes*2 + 32)
    self.convMM2 = Conv2Plus1D(planes, planes, planes*2 + 32)

class Refine3dConvTranspose(nn.Module):
  def __init__(self, inplanes, planes, scale_factor=2):
    super(Refine3dConvTranspose, self).__init__()
    self.convFS1 = nn.Conv3d(inplanes, planes, kernel_size=3, padding=1)
    self.convFS2 = nn.Conv3d(planes, planes, kernel_size=3, padding=1)
    self.convFS3 = nn.Conv3d(planes, planes, kernel_size=3, padding=1)
    self.convMM1 = nn.Conv3d(planes, planes, kernel_size=3, padding=1)
    self.convMM2 = nn.Conv3d(planes, planes, kernel_size=3, padding=1)
    # Use transpose conv to upsample the feature maps
    self.conv_t = nn.ConvTranspose3d(planes, planes, 2, stride=2, bias=True)
    self.scale_factor = scale_factor

  def forward(self, f, pm):
    s = self.convFS1(f)
    sr = self.convFS2(F.relu(s))
    sr = self.convFS3(F.relu(sr))
    s = s + sr

    m = s + F.relu(self.conv_t(pm))

    mr = self.convMM1(F.relu(m))
    mr = self.convMM2(F.relu(mr))
    m = m + mr
    return m


class Refine3dLight(Refine3d):
  def __init__(self, inplanes, planes, scale_factor=2):
    super(Refine3dLight, self).__init__(inplanes, planes, scale_factor)
    self.convFS1 = nn.Sequential( nn.Conv3d(inplanes, planes, kernel_size=1, padding=0), nn.ReLU(),
                                  nn.Conv3d(planes, planes, kernel_size=3, padding=1, groups=planes))
    self.convFS2 = nn.Sequential( nn.Conv3d(planes, planes, kernel_size=1, padding=0), nn.ReLU(),
                                  nn.Conv3d(planes, planes, kernel_size=3, padding=1, groups=planes))
    self.convFS3 = nn.Sequential( nn.Conv3d(planes, planes, kernel_size=1, padding=0), nn.ReLU(),
                                  nn.Conv3d(planes, planes, kernel_size=3, padding=1, groups=planes))

    self.convMM1 = nn.Sequential( nn.Conv3d(planes, planes, kernel_size=1, padding=0), nn.ReLU(),
                                  nn.Conv3d(planes, planes, kernel_size=3, padding=1, groups=planes))
    self.convMM2 = nn.Sequential( nn.Conv3d(planes, planes, kernel_size=1, padding=0), nn.ReLU(),
                                  nn.Conv3d(planes, planes, kernel_size=3, padding=1, groups=planes))



class Refine3dLightGN(Refine3d):
  def __init__(self, inplanes, planes, scale_factor=2, n_groups=32):
    super(Refine3dLightGN, self).__init__(inplanes, planes, scale_factor)

    def conv_gn(in_channels, out_channels):
      return nn.Sequential(
         nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
         nn.GroupNorm(n_groups, in_channels),
         nn.ReLU(inplace=True),
         nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
         nn.GroupNorm(n_groups, out_channels),
         nn.ReLU(inplace=True)
      )

    self.convFS1 = conv_gn(inplanes, planes)
    self.convFS2 = conv_gn(planes, planes)
    self.convFS3 = conv_gn(planes, planes)

    self.convMM1 = conv_gn(planes, planes)
    self.convMM2 = conv_gn(planes, planes)

  def forward(self, f, pm):
    s = self.convFS1(f)
    sr = self.convFS2(s)
    sr = self.convFS3(sr)
    s = s + sr

    m = s + F.interpolate(pm, size=s.shape[-3:], mode='trilinear')

    mr = self.convMM1(m)
    mr = self.convMM2(mr)
    m = m + mr
    return m


class UpsamplerBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.conv = nn.ConvTranspose3d(
      in_channels, out_channels, 2, stride=2, bias=True)
    # self.bn = nn.BatchNorm2d(in_channels, eps=1e-3)

  def forward(self, input):
    output = self.conv(input)
    # output = self.bn(output)
    return F.relu(output)


class Refine3dDG(nn.Module):
  def __init__(self, inplanes, planes, scale_factor=2):
    super(Refine3dDG, self).__init__()
    self.convFS1 = nn.Sequential(
      nn.Conv3d(inplanes, planes, kernel_size=1),
      nn.Conv3d(planes, planes, kernel_size=3, padding=1, groups=planes))
    self.convFS2 = nn.Sequential(
      # nn.Conv3d(planes, planes, kernel_size=1),
      nn.Conv3d(planes, planes, kernel_size=3, padding=1, groups=planes))
    self.convFS3 = nn.Sequential(
      # nn.Conv3d(planes, planes, kernel_size=1),
      nn.Conv3d(planes, planes, kernel_size=3, padding=1, groups=planes))
    self.convMM1 = nn.Sequential(
      # nn.Conv3d(planes, planes, kernel_size=1),
      nn.Conv3d(planes, planes, kernel_size=3, padding=1, groups=planes))
    self.convMM2 = nn.Sequential(
      # nn.Conv3d(planes, planes, kernel_size=1),
      nn.Conv3d(planes, planes, kernel_size=3, padding=1, groups=planes))
    self.scale_factor = scale_factor

  def forward(self, f, pm):
    s = self.convFS1(f)
    sr = self.convFS2(F.relu(s))
    sr = self.convFS3(F.relu(sr))
    s = s + sr

    m = s + F.upsample(pm, size=s.shape[-3:], mode='trilinear')

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


class NL(nn.Module):
  def __init__(self, inplanes, planes):
    super(NL, self).__init__()
    self.nl = nn.Sequential(nn.Conv3d(inplanes, planes, kernel_size=1),
                            NONLocalBlock3D(planes, sub_sample=True))

  def forward(self, x):
    x = self.nl(x)
    return x


class C3D(nn.Module):
  def __init__(self, inplanes, planes):
    super(C3D, self).__init__()
    self.c3d = nn.Conv3d(inplanes, planes, kernel_size=3, padding=1)

  def forward(self, x):
    x = self.c3d(x)
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


class PSPModule(nn.Module):
  # (1, 2, 3, 6)
  def __init__(self, sizes=(1, 3, 6, 8), dimension=2):
    super(PSPModule, self).__init__()
    self.stages = nn.ModuleList([self._make_stage(size, dimension) for size in sizes])

  def _make_stage(self, size, dimension=2):
    if dimension == 1:
      prior = nn.AdaptiveAvgPool1d(output_size=size)
    elif dimension == 2:
      prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
    elif dimension == 3:
      prior = nn.AdaptiveAvgPool3d(output_size=(size, size, size))
    return prior

  def forward(self, feats):
    n, c, _, _, _ = feats.size()
    priors = [stage(feats).view(n, c, -1) for stage in self.stages]
    center = torch.cat(priors, -1)
    return center


class _ASPPImagePooler(nn.Module):
  def __init__(self, in_planes, out_planes):
    super(_ASPPImagePooler, self).__init__()

    self.pool = nn.AdaptiveAvgPool3d((None, 1, 1))
    self.conv = nn.Conv3d(in_planes, out_planes, 1, bias=False)
    self.gn = nn.GroupNorm(16, out_planes)

  def forward(self, x):
    T, H, W = x.shape[-3:]
    x = self.pool(x)
    x = self.gn(F.relu(self.conv(x)))
    return F.interpolate(x, (T, H, W), mode='trilinear', align_corners=True)


class _ASPPConv(nn.Sequential):
  def __init__(self, in_planes, out_planes, dilation):
    super(_ASPPConv, self).__init__(
      nn.Conv3d(in_planes, in_planes, (1, 3, 3), padding=(0, dilation, dilation), dilation=(1, dilation, dilation), groups=out_planes, bias=False),
      nn.ReLU(inplace=True),
      nn.GroupNorm(16, in_planes),
      nn.Conv3d(in_planes, out_planes, 1, bias=False),
      nn.ReLU(inplace=True),
      nn.GroupNorm(16, out_planes)
    )

class ASPPModule(nn.Module):
  def __init__(self, in_planes, out_planes, inter_planes=None):
    super(ASPPModule, self).__init__()

    if not inter_planes:
      inter_planes = int(out_planes / 4)

    self.pyramid_layers = nn.ModuleList([
      nn.Sequential(
        nn.Conv3d(in_planes, inter_planes, 1, bias=False),
        nn.ReLU(inplace=True),
        nn.GroupNorm(16, inter_planes)
      ),
      _ASPPConv(in_planes, inter_planes, 3),
      _ASPPConv(in_planes, inter_planes, 6),
      _ASPPConv(in_planes, inter_planes, 9),
      _ASPPImagePooler(in_planes, inter_planes)
    ])

    self.conv = nn.Conv3d(inter_planes * 5, out_planes, 1, padding=0, bias=False)
    self.gn = nn.GroupNorm(32, out_planes)

    for m in self.modules():
      if isinstance(m, nn.Conv3d):
          n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
          m.weight.data.normal_(0, math.sqrt(2. / n))

  def forward(self, x):
    x = [layer(x) for layer in self.pyramid_layers]
    x = torch.cat(x, 1)
    return self.gn(F.relu(self.conv(x)))


class ChannelSepConv3d(nn.Sequential):
  def __init__(self, inplanes, outplanes, n_groups=32):
    super(ChannelSepConv3d, self).__init__(
      nn.Conv3d(inplanes, inplanes, 3, padding=1, groups=outplanes, bias=False),
      nn.GroupNorm(n_groups, inplanes),
      nn.ReLU(inplace=True),
      nn.Conv3d(inplanes, outplanes, 1, bias=False),
      nn.GroupNorm(n_groups, outplanes),
      nn.ReLU(inplace=True)
    )


class BMVC19Decoder(nn.Module):
  def __init__(self):
    super().__init__()

    self.conv1 = ChannelSepConv3d(512, 128)
    self.conv2 = ChannelSepConv3d(256, 128)
    self.conv3 = ChannelSepConv3d(256, 64)

    self.conv_a = ChannelSepConv3d(128, 128)
    self.conv_b = ChannelSepConv3d(64, 128)

    self.conv_out = nn.Conv3d(64, 2, 1, bias=False)

  def forward(self, x):
    x3, x2, x1 = x  # largest to smallest in size

    x = self.conv1(x1)
    x = F.interpolate(x, x2.shape[-3:], mode='trilinear', align_corners=True)
    x = torch.cat((x, x2), 1)

    x = self.conv2(x)
    x = F.interpolate(x, x3.shape[-3:], mode='trilinear', align_corners=True)
    x = torch.cat((x, x3), 1)

    x = F.interpolate(x, scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)
    return self.conv_out(x)


def test_aspp():
  aspp = ASPPModule(256, 64, 256).cuda()
  x = torch.zeros(1, 256, 1, 120, 210, dtype=torch.float32).cuda()

  y = aspp(x)
  print(y.shape)


def test_bmvc19decoder():
  x = [
    torch.zeros(1, 64, 8, 240, 427, dtype=torch.float32).cuda(),
    torch.zeros(1, 128, 8, 120, 214, dtype=torch.float32).cuda(),
    torch.zeros(1, 512, 4, 30, 54, dtype=torch.float32).cuda()
  ]

  decoder = BMVC19Decoder().cuda()
  print(decoder(x).shape)


if __name__ == '__main__':
  # test_aspp()
  test_bmvc19decoder()

