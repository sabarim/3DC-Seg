import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# from train import show_image_summary
from network.NetworkUtil import run_forward
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
    all_preds = None
    for i, input_dict in enumerate(dataloader):
      with torch.no_grad():
        info = input_dict['info']
        if all_preds is None:
          all_preds = torch.zeros((info['num_frames'], info['num_objects'],) +
                                  (tuple(input_dict['masks_guidance'].shape[-2:],)))
          all_targets = torch.zeros((info['num_frames'],1,) +
                                  (tuple(input_dict['masks_guidance'].shape[-2:], )))
          for object in range(info['num_objects']):
            all_preds[0, object] = (input_dict['target'] == object+1).float()
          all_targets[0] = input_dict["target"]
          continue

        input_dict = input_dict.copy()
        input_dict['masks_guidance'] = all_preds[info['support_indices'][0]].unsqueeze(0)
        # TODO: remove this after testing
        # for object in range(info['num_objects']):
        #   input_dict['masks_guidance'][0, -3, object] = (all_targets[info['support_indices'][0, -3]] == object+1).float()[0]
        # input_dict['masks_guidance'][0, 1, object] = \
        # (all_targets[info['support_indices'][0, 1]] == object + 1).float()[0]
        # compute output
        input, input_var, iou, loss, loss_image, masks_guidance, output, target = forward(criterion, input_dict,
                                                                                                  model)
        all_preds[i] = output
        all_targets[i] = target
        # all_preds[i] = target.repeat(1,info['num_objects'],1,1)
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
  preds = []
  for object in range(info['num_objects']):
    # data_time.update(time.time() - end)
    # input_guidance = (masks_guidance == object+1).float().cuda()
    input_guidance = masks_guidance[:, :, object].unsqueeze(0)
    input_var = input.float().cuda()

    label = (target == object+1).float()
    # compute output
    pred = run_forward(model, input_var, input_guidance, None)[0]
    preds += [torch.argmax(pred, dim=1)]
    loss_image = criterion(pred[:, -1], label.squeeze(1).cuda().float())
    loss = bootstrapped_ce_loss(loss_image)

    iou = iou_fixed(pred.data.cpu().numpy(), label.data.cpu().numpy())
    loss_object += [loss.data.item()]
    iou_object += [iou]
  return input, input_var, np.mean(iou_object), np.mean(loss_object), loss_image, masks_guidance, \
         torch.cat(preds, dim=0), target


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


def remove_padding(tensor, info):
  (lh, uh), (lw, uw) = info['pad']
  E = tensor[:, :, lh[0]:-uh[0], lw[0]:-uw[0]]
  return E
