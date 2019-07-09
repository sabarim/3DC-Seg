import time
import torch
import numpy as np
from torch.nn import functional as F

from utils.AverageMeter import AverageMeter
from utils.Loss import bootstrapped_ce_loss
from utils.util import iou_fixed


def infer_DAVIS(dataloader, model, criterion, writer):
  batch_time = AverageMeter()
  losses = AverageMeter()
  ious = AverageMeter()

  # switch to evaluate mode
  model.eval()

  end = time.time()
  for seq in dataloader.dataset.get_video_ids():
    dataloader.dataset.set_video_id(seq)
    ious_video = AverageMeter()
    for i, input_dict in enumerate(dataloader):
      with torch.no_grad():
        # compute output
        input, input_var, iou, loss, loss_image, masks_guidance, output, target = forward(criterion, input_dict,
                                                                                          model)
        ious_video.update(iou)
        losses.update(loss)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print('Test: {0} [{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.5f})\t'
              'IOU {iou.val:.4f} ({iou.avg:.5f})\t'.format(
          input_dict['info']['name'], i, len(dataloader), batch_time=batch_time, loss=losses, iou=ious_video))
    print('Sequence {}\t IOU {iou.avg}'.format(input_dict['info']['name'], iou=ious_video))
    ious.update(ious_video.avg)

  print('Finished Inference Loss {losses.avg:.5f} IOU {iou.avg: 5f}'
        .format(losses=losses, iou=ious))

  return losses.avg, ious.avg

def forward(criterion, input_dict, model):
  input = input_dict["images"]
  target = input_dict["target"]
  masks_guidance = input_dict["masks_guidance"]
  info = input_dict["info"]
  shape = info['shape']
  iou_object = []
  loss_object = []
  for object in range(info['num_objects']):
    # data_time.update(time.time() - end)
    input_guidance = (masks_guidance == object+1).float().cuda()
    input_var = input.float().cuda()

    label = remove_padding((target == object+1).float(), info)
    label = F.interpolate(label, size=shape)
    # compute output
    pred = propagate(model, input_var, input_guidance)
    pred = F.interpolate(pred[0], target.shape[2:], mode="bilinear")

    pred = remove_padding(pred, info)
    pred = F.interpolate(pred, size=shape)
    loss_image = criterion(pred[:, -1], label.squeeze(1).cuda().float())
    loss = bootstrapped_ce_loss(loss_image)

    iou = iou_fixed(pred.data.cpu().numpy(), label.data.cpu().numpy())
    loss_object += [loss.data.item()]
    iou_object += [iou]
  return input, input_var, np.mean(iou_object), np.mean(loss_object), loss_image, masks_guidance, pred, target


def propagate(model, inputs, ref_mask):
  refs = []
  assert inputs.shape[2] >= 2
  for i in range(inputs.shape[2]):
    r5, r4, r3, r2 = model.encoder(inputs[:, :, i], ref_mask[:, :, i])
    refs+=[r5.unsqueeze(2)]
  support = torch.cat(refs[:-1], dim=2)
  e2 = model.decoder(r5, r4, r3, r2, support)

  return (F.softmax(e2[0], dim=1), r5, e2[-1])


def remove_padding(tensor, info):
  (lh, uh), (lw, uw) = info['pad']
  E = tensor[:, :, lh[0]:-uh[0], lw[0]:-uw[0]]
  return E