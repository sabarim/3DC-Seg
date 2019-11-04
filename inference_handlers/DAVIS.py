import os
import time

import numpy as np
import torch
from PIL import Image
from scipy.misc import imresize
from torch import nn
from torch.nn import functional as F

from loss.Loss import bootstrapped_ce_loss
from network.NetworkUtil import run_forward
from util import create_object_id_mapping, get_one_hot_vectors
from utils.AverageMeter import AverageMeter
from utils.util import iou_fixed, get_iou

palette = Image.fromarray(np.zeros((480,854))).getpalette()
IOU_THRESH = 0.1


def infer_DAVIS(dataloader, model, criterion, writer, args):
  batch_time = AverageMeter()
  losses = AverageMeter()
  ious = AverageMeter()
  results_path = os.path.join("results/", args.network_name)
  # switch to evaluate mode
  model.eval()

  end = time.time()
  iter = 0
  for seq in dataloader.dataset.get_video_ids():
  # for seq in ['goat']:
    dataloader.dataset.set_video_id(seq)
    ious_video = AverageMeter()
    all_preds = None
    obj_categories = []
    for i, input_dict in enumerate(dataloader):
      with torch.no_grad():
        info = input_dict['info']
        if all_preds is None:
          all_preds = torch.zeros((info['num_frames'], info['num_objects'],) +
                                  (tuple(input_dict['masks_guidance'].shape[-2:],)))
          all_targets = torch.zeros((info['num_frames'],1,) +
                                  (tuple(input_dict['masks_guidance'].shape[-2:], )))
          object_mapping = create_object_id_mapping(input_dict['target'][0,0].data.cpu().numpy(),
                                                    get_one_hot_vectors(input_dict['proposals'][0, 0, 0].data.cpu().numpy()))
          # obj_cats = input_dict['raw_proposals']['labels'][0][list(object_mapping.values())]
          for key in object_mapping.keys():
            object_mapping[key] = input_dict['raw_proposals']['labels'][0][object_mapping[key]]
          for obj_id in object_mapping.keys():
            all_preds[0, obj_id-1] = (input_dict['target'][0,0] == obj_id).float()
          all_targets[0] = input_dict["target"]
          continue

        input_dict = input_dict.copy()
        input_dict['masks_guidance'] = all_preds[info['support_indices'][0]].unsqueeze(0)
        input, input_var, iou, loss, loss_image, masks_guidance, output, target = forward(criterion, input_dict,
                                                                                                  model, obj_categories,
                                                                                          writer, iter,
                                                                                          args.show_image_summary)
        iter += info['num_objects'][0]
        if args.save_results:
          pred = torch.cat(((torch.sum(output, dim=0) == 0).unsqueeze(0).float(),
                                output))
          save_results(pred, info, i, results_path)

        # for key in object_mapping.keys():
        #   val = object_mapping[key]
        #   keep = (input_dict['raw_proposals']['labels'] == val)
        #   proposals_filtered = get_one_hot_vectors(input_dict['proposals'][0, 0, -1].data.cpu().numpy())[np.array(keep[0])]
        #   background =(proposals_filtered.sum(axis=0)==0).astype(np.int)
        #   all_preds[i, key-1] = torch.argmax(torch.cat((torch.from_numpy(background.astype(np.float32)).unsqueeze(0),
        #                                                 torch.from_numpy(proposals_filtered.astype(np.float32))), dim=0),
        #                                      dim=0)

        all_preds[i] = split_proposals(input_dict, object_mapping)
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


def forward(criterion, input_dict, model, obj_cats, writer, iter, image_summary):
  input = input_dict["images"]
  target = input_dict["target"]
  proposals = input_dict['proposals']
  masks_guidance = input_dict["masks_guidance"]
  info = input_dict["info"]
  shape = info['shape']
  iou_object = []
  loss_object = []
  preds = []
  # assume a batch size of 1 during inference
  assert input.shape[0] == 1
  for object in range(info['num_objects']):
    # data_time.update(time.time() - end)
    # input_guidance = (masks_guidance == object+1).float().cuda()
    input_guidance = masks_guidance[:, :, object].unsqueeze(0)
    input_var = input.float().cuda()

    label = (target == object+1).float()
    # compute output
    logits = run_forward(model, input_var, input_guidance, proposals)[0]
    pred = torch.argmax(F.softmax(logits, dim=1), dim=1)
    # store the logits for foreground region to make decisions in the regions where objects overlap
    fg_logits = logits[:, -1] * pred.float()
    preds += [pred.float()]
    if image_summary:
      show_image_summary(iter, writer, input_var, input_guidance, label, pred)

    loss_image = criterion(logits[:, -1], label.squeeze(1).cuda().float())
    loss = bootstrapped_ce_loss(loss_image)

    iou = iou_fixed(logits.data.cpu().numpy(), label.data.cpu().numpy())
    loss_object += [loss.data.item()]
    iou_object += [iou]
    iter += 1

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


def save_results(pred, info, f, results_path):
  results_path = os.path.join(results_path, info['name'][0])
  E = pred.data.cpu().numpy()
  # make hard label
  E = np.argmax(E, axis=0).astype(np.uint8)

  (lh, uh), (lw, uw) = info['pad']
  E = E[lh[0]:-uh[0], lw[0]:-uw[0]]


  img_E = Image.fromarray(imresize(E, info['shape'], interp='nearest'))
  img_E.putpalette(palette)
  if not os.path.exists(results_path):
    os.makedirs(results_path)
  img_E.save(os.path.join(results_path, '{:05d}.png'.format(f)))


def get_best_overlap(ref_obj, proposals, cats, obj_cat=None):
  best_iou = 0
  target_id = -1
  mask = proposals[:, 0].cuda()
  obj_cat = None if obj_cat == -1 else obj_cat

  for obj_id in proposals.unique():
    if obj_id == 0:
      continue
    iou = get_iou(ref_obj[0].int().data.cpu().numpy(), (proposals[0,0] == obj_id).data.cpu().numpy().astype(np.uint8))
    if iou > best_iou and iou > IOU_THRESH and (obj_cat is None or cats[obj_id.int()] == obj_cat):
      best_iou = iou
      target_id = cats[obj_id.int()]
      mask = (proposals[:, 0] == obj_id).int().cuda()

  return mask, best_iou, target_id


def show_image_summary(count, foo, input_var, masks_guidance, target, pred):
  for index in range(input_var.shape[2]):
    foo.add_images("data/input" + str(index), input_var[:, :3, index], count)
    foo.add_images("data/guidance" + str(index), masks_guidance[:, :, index].repeat(1, 3, 1, 1), count)
  # foo.add_image("data/loss_image", loss_image.unsqueeze(1), count)
  foo.add_images("data/target", target.repeat(1,3,1,1), count)
  foo.add_images("data/pred", pred.unsqueeze(1).repeat(1,3,1,1), count)


def split_proposals(input_dict, object_mapping):
  preds = torch.zeros((len(object_mapping.keys()),) + (input_dict['proposals'].shape[-2:]))
  for key in object_mapping.keys():
    val = object_mapping[key]
    keep = (input_dict['raw_proposals']['labels'][0] == val)
    # if no proposals are selected, then use all the proposals
    if keep.sum() == 0:
      keep = torch.ones_like(keep).bool()
    proposals_filtered = get_one_hot(input_dict['proposals'][0, 0, -1].data.cpu().numpy(), keep)[np.array(keep)]
    background = (proposals_filtered.sum(axis=0) == 0).astype(np.int)
    preds[key-1] = torch.argmax(torch.cat((torch.from_numpy(background.astype(np.float32)).unsqueeze(0),
                                                    torch.from_numpy(proposals_filtered.astype(np.float32))), dim=0),
                                         dim=0)
  return preds


def get_one_hot(mask, cats):
  num_objects = np.setdiff1d(np.unique(mask), [0])
  one_hot_mask = np.zeros((len(cats),) + mask.shape)

  for i in num_objects:
    one_hot_mask[i-1] = (mask == i).astype(np.uint8)

  return one_hot_mask

