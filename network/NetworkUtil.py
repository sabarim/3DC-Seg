import torch
from torch import nn
from torch.nn import functional as F


def propagate(model, inputs, ref_mask):
  refs = []
  assert inputs.shape[2] >= 2
  for i in range(inputs.shape[2]):
    encoder = nn.DataParallel(model.module.encoder)
    r5, r4, r3, r2 = encoder(inputs[:, :, i], ref_mask[:, :, i])
    refs+=[r5.unsqueeze(2)]
  support = torch.cat(refs[:-1], dim=2)
  decoder = nn.DataParallel(model.module.decoder)
  e2 = decoder(r5, r4, r3, r2, support)

  return (F.softmax(e2[0], dim=1), r5, e2[-1])


def propagateMultiEncoder(model, inputs, ref_mask, proposals):
  refs = []
  assert inputs.shape[2] >= 2
  for i in range(inputs.shape[2] - 1):
    encoder = nn.DataParallel(model.module.encoder)
    r5, r4, r3, r2 = encoder(inputs[:, :, i], ref_mask[:, :, i])
    refs+=[r5.unsqueeze(2)]
  # forward the current mask with a separate encoder
  encoderCurr = nn.DataParallel(model.module.encoder2)
  # r5, r4, r3, r2 = encoderCurr(inputs[:, :, -1], proposals[:, :, -1])
  r5, r4, r3, r2 = encoderCurr(inputs[:, :, -1], None)
  support = torch.cat(refs, dim=2)
  decoder = nn.DataParallel(model.module.decoder)
  e2 = decoder(r5, r4, r3, r2, support)

  return (F.softmax(e2[0], dim=1), r5, e2[-1])


def propagate3d(model, inputs, ref_mask, proposals):
  refs = []
  assert inputs.shape[2] >= 2
  e2 = model(inputs, ref_mask)

  return (F.softmax(e2[0], dim=1), e2[-1], e2[-2])


def run_forward(model, inputs, ref_masks, proposals):
  if 'multi' in str(model.module.__class__).lower():
    return propagateMultiEncoder(model, inputs, ref_masks, proposals)
  elif 'resnet3d' in str(model.module.__class__).lower():
    return propagate3d(model, inputs, ref_masks, proposals)
  else:
    return propagate(model, inputs, ref_masks)