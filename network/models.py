import torch
from torch import nn
from torch.nn import functional as F

from network.NetworkUtil import get_backbone_fn, get_module
from utils import Constants


class BaseNetwork(nn.Module):
  def __init__(self, tw=5):
    super(BaseNetwork, self).__init__()
    self.tw = tw


class Encoder3d(nn.Module):
  def __init__(self, backbone, tw, pixel_mean, pixel_std):
    super(Encoder3d, self).__init__()
    self.conv1_p = nn.Conv3d(1, 64, kernel_size=7, stride=(1, 2, 2),
                             padding=(3, 3, 3), bias=False)

    resnet = get_backbone_fn(backbone.NAME)(sample_size=112, sample_duration = tw)
    if backbone.PRETRAINED_WTS:
      print('Loading pretrained weights for the backbone from {} {}{}...'.format(Constants.font.BOLD,
                                                                                 backbone.PRETRAINED_WTS, Constants.font.END))
      chkpt = torch.load(backbone.PRETRAINED_WTS)
      resnet.load_state_dict(chkpt)

    self.resnet = resnet

    self.resnet = resnet
    self.conv1 = resnet.conv1
    self.bn1 = resnet.bn1
    self.relu = resnet.relu  # 1/2, 64
    self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

    self.layer1 = resnet.layer1  # 1/4, 256
    self.layer2 = resnet.layer2  # 1/8, 512
    self.layer3 = resnet.layer3  # 1/16, 1024
    self.layer4 = resnet.layer4  # 1/32, 2048

    self.register_buffer('mean', torch.FloatTensor(pixel_mean).view(1, 3, 1, 1, 1))
    self.register_buffer('std', torch.FloatTensor(pixel_std).view(1, 3, 1, 1, 1))

    if backbone.FREEZE_BN:
      self.freeze_batchnorm()

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


class Decoder3d(nn.Module):
  def __init__(self, n_classes, inter_block, refine_block, pred_scale_factor=(1,4,4)):
    super(Decoder3d, self).__init__()
    mdim = 256
    self.pred_scale_factor = pred_scale_factor
    self.GC = get_module(inter_block)(2048, mdim)
    self.convG1 = nn.Conv3d(mdim, mdim, kernel_size=3, padding=1)
    self.convG2 = nn.Conv3d(mdim, mdim, kernel_size=3, padding=1)
    refine_cls = get_module(refine_block)
    self.RF4 = refine_cls(1024, mdim)  # 1/16 -> 1/8
    self.RF3 = refine_cls(512, mdim)  # 1/8 -> 1/4
    self.RF2 = refine_cls(256, mdim)  # 1/4 -> 1

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


class SaliencyNetwork(BaseNetwork):
  def __init__(self, cfg):
    super(SaliencyNetwork, self).__init__()
    self.encoder = Encoder3d(cfg.MODEL.BACKBONE, cfg.INPUT.TW, cfg.MODEL.PIXEL_MEAN, cfg.MODEL.PIXEL_STD)
    self.decoder = Decoder3d(cfg.MODEL.N_CLASSES, inter_block=cfg.MODEL.DECODER.INTER_BLOCK,
                             refine_block=cfg.MODEL.DECODER.REFINE_BLOCK)
    if cfg.MODEL.FREEZE_BN:
      self.encoder.freeze_batchnorm()

  def forward(self, x, ref=None):
    if ref is not None and len(ref.shape) == 4:
      r5, r4, r3, r2 = self.encoder.forward(x, ref.unsqueeze(2))
    else:
      r5, r4, r3, r2 = self.encoder.forward(x, ref)
    p = self.decoder.forward(r5, r4, r3, r2, None)
    return [p]
