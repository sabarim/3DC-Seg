import inspect

# def propagate(model, inputs, ref_mask):
#   refs = []
#   assert inputs.shape[2] >= 2
#   for i in range(inputs.shape[2]):
#     encoder = nn.DataParallel(model.module.encoder)
#     r5, r4, r3, r2 = encoder(inputs[:, :, i], ref_mask[:, :, i])
#     refs+=[r5.unsqueeze(2)]
#   support = torch.cat(refs[:-1], dim=2)
#   decoder = nn.DataParallel(model.module.decoder)
#   e2 = decoder(r5, r4, r3, r2, support)
#
#   return (e2[0], r5, e2[-1])
#
#
# def propagateMultiEncoder(model, inputs, ref_mask, proposals):
#   refs = []
#   assert inputs.shape[2] >= 2
#   for i in range(inputs.shape[2] - 1):
#     encoder = nn.DataParallel(model.module.encoder)
#     r5, r4, r3, r2 = encoder(inputs[:, :, i], ref_mask[:, :, i])
#     refs+=[r5.unsqueeze(2)]
#   # forward the current mask with a separate encoder
#   encoderCurr = nn.DataParallel(model.module.encoder2)
#   # r5, r4, r3, r2 = encoderCurr(inputs[:, :, -1], proposals[:, :, -1])
#   r5, r4, r3, r2 = encoderCurr(inputs[:, :, -1], None)
#   support = torch.cat(refs, dim=2)
#   decoder = nn.DataParallel(model.module.decoder)
#   e2 = decoder(r5, r4, r3, r2, support)
#
#   return (e2[0], r5, e2[-1])
from network import Resnet3d, Modules


def propagate3d(model, inputs, ref_mask, proposals):
  assert inputs.shape[2] >= 2
  e2 = model(inputs, ref_mask)

  return e2


def run_forward(model, inputs, ref_masks, proposals):
  return propagate3d(model, inputs, ref_masks, proposals)


def get_backbone_fn(backbone):
  """
  Returns a funtion that creates the required backbone
  :param backbone: name of the backbone function
  :return:
  """
  backbones = inspect.getmembers(Resnet3d)
  _fn = [_f for (name, _f) in backbones if name == backbone]
  if len(_fn) == 0:
    raise ValueError("Backbone {} can't be found".format(backbone))
  return _fn[0]


def get_module(module):
  backbones = inspect.getmembers(Modules)
  _cls = [_c for (name, _c) in backbones if name == module]
  return _cls[0]

