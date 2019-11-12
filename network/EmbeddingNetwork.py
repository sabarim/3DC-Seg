import torch
from torch import nn

from network.Modules import Refine3dDG
from network.Resnet3d import resnet50
from network.Resnet3dAgg import Encoder3d, Decoder3d, Resnet3dSimilarity
from network.embedding_head import NonlocalOffsetEmbeddingHead
from network.models import BaseNetwork
from torch.nn import functional as F


class DecoderWithEmbedding(Decoder3d):
  def __init__(self, n_classes=2, e_dim = 64, add_spatial_coord=True):
    super(DecoderWithEmbedding, self).__init__(n_classes)
    self.embedding_head = NonlocalOffsetEmbeddingHead(256, 128, e_dim, downsampling_factor=2, add_spatial_coord=add_spatial_coord)

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


class DecoderEmbedding(DecoderWithEmbedding):
  def __init__(self,  n_classes=2, e_dim = 3, add_spatial_coord=True):
    super(DecoderEmbedding, self).__init__( n_classes=n_classes, e_dim = e_dim, add_spatial_coord=add_spatial_coord)
    self.convG1 = nn.Conv3d(2048, 1024, kernel_size=3, padding=1)
    self.convG2 = nn.Conv3d(1024, 256, kernel_size=3, padding=1)
    # self.RF4 = Refine3dDG(1024, 256)
    # self.RF3 = Refine3dDG(512, 256)
    #self.fp16Refinement()

  def fp16Refinement(self):
    for layer in self.modules():
      if isinstance(layer, Refine3dDG):
        layer.half()

  def forward(self, r5, r4, r3, r2, support):
    x = self.convG1(r5)
    x = self.convG2(F.relu(x))
    m4 = self.RF4(r4, F.relu(x))
    m3 = self.RF3(r3, m4)
    e = self.embedding_head(F.relu(m3))

    return e


class Resnet3dEmbeddingNetwork(Resnet3dSimilarity):
  def __init__(self, tw=8, sample_size=112, e_dim=64):
    super(Resnet3dEmbeddingNetwork, self).__init__()
    self.encoder = Encoder3d(tw, sample_size)
    self.decoder = DecoderWithEmbedding(e_dim=e_dim)


class Resnet3dSegmentEmbedding(Resnet3dSimilarity):
  def __init__(self, tw=8, sample_size=112,n_classes=2, e_dim=64):
    super(Resnet3dSegmentEmbedding, self).__init__(n_classes=n_classes)
    self.encoder = Encoder3d(tw, sample_size)
    self.decoder = DecoderSegmentEmbedding(n_classes=n_classes, e_dim=e_dim)


class Resnet3dSpatialEmbedding(Resnet3dSimilarity):
  def __init__(self, tw=8, sample_size=112,n_classes=2, e_dim=64):
    super(Resnet3dSpatialEmbedding, self).__init__(n_classes=n_classes)
    self.encoder = Encoder3d(tw, sample_size)
    self.decoder = DecoderWithEmbedding(e_dim=e_dim, add_spatial_coord=False)


class Resnet3dEmbeddingMultiDecoder(Resnet3dSimilarity):
  def __init__(self, tw=8, sample_size=112,n_classes=2, e_dim=64):
    super(Resnet3dEmbeddingMultiDecoder, self).__init__(n_classes=n_classes)
    resnet = resnet50(sample_size=sample_size, sample_duration=tw)
    self.encoder = Encoder3d(tw, sample_size, resnet=resnet)
    self.decoder = Decoder3d()
    self.decoder_embedding = DecoderEmbedding(e_dim=e_dim, add_spatial_coord=False)

  def forward(self, x, ref = None):
    r5, r4, r3, r2 = self.encoder.forward(x, ref)
    p = self.decoder.forward(r5, r4, r3, r2, None)
    e = self.decoder_embedding.forward(r5, r4, r3, r2, None)
    return [p,e]


