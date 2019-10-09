from network.Resnet3dAgg import Encoder3d, Decoder3d, Resnet3dSimilarity
from network.embedding_head import NonlocalOffsetEmbeddingHead
from network.models import BaseNetwork
from torch.nn import functional as F


class DecoderWithEmbedding(Decoder3d):
  def __init__(self):
    super(DecoderWithEmbedding, self).__init__()
    self.embedding_head = NonlocalOffsetEmbeddingHead(256, 128, 64, downsampling_factor=2)

  def forward(self, r5, r4, r3, r2, support):
    x = self.GC(r5)
    r = self.convG1(F.relu(x))
    r = self.convG2(F.relu(r))
    m5 = x + r  # out: 1/32, 64
    m4 = self.RF4(r4, m5)  # out: 1/16, 64
    m3 = self.RF3(r3, m4)  # out: 1/8, 64
    m2 = self.RF2(r2, m3)  # out: 1/4, 64
    e = self.embedding_head(m3)

    p2 = self.pred2(F.relu(m2))
    p3 = self.pred3(F.relu(m3))
    p4 = self.pred4(F.relu(m4))
    p5 = self.pred5(F.relu(m5))

    p = F.interpolate(p2, scale_factor=(1, 4, 4), mode='trilinear')

    return p, e


class Resnet3dEmbeddingNetwork(Resnet3dSimilarity):
  def __init__(self, tw=8, sample_size=112):
    super(Resnet3dEmbeddingNetwork, self).__init__()
    self.encoder = Encoder3d(tw, sample_size)
    self.decoder = DecoderWithEmbedding()


