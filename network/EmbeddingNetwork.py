from functools import reduce
from operator import add

import torch
from torch import nn
from torch.nn import functional as F

from network.Modules import Refine3dConvTranspose, Refine3dLight
from network.Resnet3d import resnet50_no_ts
from network.Resnet3dAgg import Encoder3d, Decoder3d, Encoder3d_csn_ir, Resnet3d
from network.embedding_head import NonlocalOffsetEmbeddingHead
from network.modules.multiscale import MultiscaleCombinedHeadLongTemporalWindow


class DecoderWithEmbedding(Decoder3d):
  def __init__(self, n_classes=2, e_dim = 64, add_spatial_coord=True):
    super(DecoderWithEmbedding, self).__init__(n_classes)
    self.embedding_head = NonlocalOffsetEmbeddingHead(256, 128, e_dim, downsampling_factor=2,
                                                      add_spatial_coord=add_spatial_coord)

  def forward(self, r5, r4, r3, r2, support):
    x = self.GC(r5)
    r = self.convG1(F.relu(x))
    r = self.convG2(F.relu(r))
    m5 = x + r  # out: 1/32, 64
    m4 = self.RF4(r4, m5)  # out: 1/16, 64
    m3 = self.RF3(r3, m4)  # out: 1/8, 64
    m2 = self.RF2(r2, m3)  # out: 1/4, 64
    e = self.embedding_head(F.interpolate(F.relu(m2), scale_factor=(1,0.5,0.5), mode='trilinear'))

    p2 = self.pred2(F.relu(m2))
    p = F.interpolate(p2, scale_factor=(1, 4, 4), mode='trilinear')

    return p, e, m2


class DecoderSegmentEmbedding(DecoderWithEmbedding):
  def __init__(self, n_classes=2, e_dim=64):
    super(DecoderSegmentEmbedding, self).__init__(n_classes=n_classes, e_dim=e_dim)
    # self.convG1 = nn.Conv3d(2048, 256, kernel_size=3, padding=1)
    self.con1x1 = nn.Conv3d(e_dim, 256, kernel_size=1, padding=1)

  def forward(self, r5, r4, r3, r2, support):
    x = self.GC(r5)
    r = self.convG1(F.relu(x))
    r = self.convG2(F.relu(r))
    m5 = x + r  # out: 1/32, 64
    m4 = self.RF4(r4, m5)  # out: 1/16, 64
    m3 = self.RF3(r3, m4)  # out: 1/8, 64
    m2 = self.RF2(r2, m3)  # out: 1/4, 64
    e = self.embedding_head(F.interpolate(m3, scale_factor=(2, 1, 1), mode='trilinear'))

    e_unrolled = self.con1x1(F.relu(e))
    p2 = self.pred2(F.relu(m2) + F.interpolate(e_unrolled, m2.shape[2:], mode='trilinear'))
    p = F.interpolate(p2, scale_factor=(1, 4, 4), mode='trilinear')

    return p, e, m2


class DecoderEmbedding(Decoder3d):
  def __init__(self,  n_classes=2, e_dim = 3, add_spatial_coord=True, scale=0.5):
    super(DecoderEmbedding, self).__init__( n_classes=n_classes)
    self.RF4 = Refine3dConvTranspose(1024, 256)
    self.RF3 = Refine3dConvTranspose(512, 256)
    self.RF2 = Refine3dConvTranspose(256, 256)
    # self.pred2 = nn.Conv3d(256, n_classes, kernel_size=3, padding=1, stride=1, bias=False)


class DecoderLight(Decoder3d):
  def __init__(self, n_classes=2, conv_t = False):
    super(DecoderLight, self).__init__(n_classes=n_classes)
    self.RF4 = Refine3dLight(1024, 256, conv_t=conv_t)
    self.RF3 = Refine3dLight(512, 256, conv_t=conv_t)
    self.RF2 = Refine3dLight(256, 256, conv_t=conv_t)


class DecoderMultiClass(Decoder3d):
  def __init__(self, n_classes=2, conv_t=False):
    super(DecoderMultiClass, self).__init__(n_classes=n_classes)
    self.pred_fg = nn.Conv3d(256, 2, kernel_size=3, padding=1, stride=1)

  def forward(self, r5, r4, r3, r2, support):
    x = self.GC(r5)
    r = self.convG1(F.relu(x))
    r = self.convG2(F.relu(r))
    m5 = x + r  # out: 1/32, 64
    m4 = self.RF4(r4, m5)  # out: 1/16, 64
    m3 = self.RF3(r3, m4)  # out: 1/8, 64
    m2 = self.RF2(r2, m3)  # out: 1/4, 64

    p_multi = self.pred2(F.relu(m2))
    p_fg = self.pred_fg(F.relu(m2))
    p_multi = F.interpolate(p_multi, scale_factor=(1, 4, 4), mode='trilinear')
    p_fg = F.interpolate(p_fg, scale_factor=(1, 4, 4), mode='trilinear')
    
    # p = torch.cat((p2, p_fg[:, -1:]), dim=1)
    p = [p_fg, p_multi]

    return p


# Multi scale decoder
class MultiScaleDecoder(Decoder3d):
  def __init__(self, n_classes=2, add_spatial_coord = True):
    super(MultiScaleDecoder, self).__init__(n_classes)
    self.convG1 = nn.Conv3d(2048, 256, kernel_size=3, padding=1)
    self.embedding_head = MultiscaleCombinedHeadLongTemporalWindow(256, n_classes,True, True,seed_map=True,
                                                                   add_spatial_coord=add_spatial_coord)

  def forward(self, r5, r4, r3, r2, support):
    r = self.convG1(F.relu(r5))
    r = self.convG2(F.relu(r))
    m5 = r  # out: 1/32, 64
    m4 = self.RF4(r4, m5)  # out: 1/16, 64
    m3 = self.RF3(r3, m4)  # out: 1/8, 64
    m2 = self.RF2(r2, m3)  # out: 1/4, 64
    p, e = self.embedding_head.forward([m5, m4, m3, m2])

    return p, e


class Resnet3dEmbeddingMultiDecoder(Resnet3d):
  def __init__(self, tw=8, sample_size=112, e_dim=7, decoders=None):
    super(Resnet3dEmbeddingMultiDecoder, self).__init__(tw=tw, sample_size=sample_size)
    resnet = resnet50_no_ts(sample_size=sample_size, sample_duration=tw)
    self.encoder = Encoder3d(tw, sample_size, resnet=resnet)
    decoders = [Decoder3d(), DecoderEmbedding(n_classes=e_dim)] if decoders is None else decoders
    self.decoders = nn.ModuleList()
    for decoder in decoders:
      self.decoders.append(decoder)

  def forward(self, x, ref = None):
    r5, r4, r3, r2 = self.encoder.forward(x, ref)
    flatten = lambda lst: [lst] if type(lst) is torch.Tensor else reduce(add, [flatten(ele) for ele in lst])
    p = flatten([decoder.forward(r5, r4, r3, r2, None) for decoder in self.decoders])
    # e = self.decoder_embedding.forward(r5, r4, r3, r2, None)
    return p


class Resnet3dChannelSeparated_ir(Resnet3dEmbeddingMultiDecoder):
  def __init__(self, tw=16, sample_size = 112, e_dim=7, n_classes=2, decoders=None):
    decoders = [Decoder3d(n_classes=n_classes), DecoderEmbedding(n_classes=e_dim)] if decoders is None else decoders
    super(Resnet3dChannelSeparated_ir, self).__init__( decoders =decoders)
    self.encoder = Encoder3d_csn_ir(tw, sample_size)


class Resnet3dCSNiRSameDecoders(Resnet3dEmbeddingMultiDecoder):
  def __init__(self, tw=16, sample_size = 112, e_dim=7):
    super(Resnet3dCSNiRSameDecoders, self).__init__(decoders=
                                                      [Decoder3d(),
                                                       Decoder3d(n_classes=e_dim)
                                                       ])
    self.encoder = Encoder3d_csn_ir(tw, sample_size)


class Resnet3dCSNiRLight(Resnet3dEmbeddingMultiDecoder):
  def __init__(self, tw=16, sample_size = 112, e_dim=7):
    super(Resnet3dCSNiRLight, self).__init__(decoders=
                                                      [DecoderLight(),
                                                       DecoderLight(n_classes=e_dim, conv_t=True)
                                                       ])
    self.encoder = Encoder3d_csn_ir(tw, sample_size)


class Resnet3dCSNiRMultiScale(Resnet3d):
  def __init__(self, tw=16, sample_size = 112, e_dim=7, add_spatial_coord=True):
    super(Resnet3dCSNiRMultiScale, self).__init__()
    self.encoder = Encoder3d_csn_ir(tw, sample_size)
    self.decoder = MultiScaleDecoder(add_spatial_coord=add_spatial_coord)

  def forward(self, x, ref):
    r5, r4, r3, r2 = self.encoder.forward(x, ref)
    p = self.decoder.forward(r5, r4, r3, r2, None)

    return p


class Resnet3dCSNiRMultiClass(Resnet3dChannelSeparated_ir):
  def __init__(self, tw=16, sample_size = 112, e_dim=7, n_classes=2):
    super(Resnet3dCSNiRMultiClass, self).__init__(tw=tw, sample_size=sample_size, e_dim=e_dim, n_classes=n_classes,
                                                  decoders=[DecoderMultiClass(n_classes=n_classes),
                                                            DecoderEmbedding(n_classes=e_dim)])
