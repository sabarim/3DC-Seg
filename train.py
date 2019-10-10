import os
import time

import numpy as np
import torch
from PIL import Image
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler

from inference_handlers.inference import infer
from network.NetworkUtil import run_forward
from network.RGMP import Encoder
from utils.Argparser import parse_args
# Constants
from utils.AverageMeter import AverageMeter
from utils.Constants import DAVIS_ROOT
from utils.Loss import bootstrapped_ce_loss
from utils.Saver import load_weights, save_checkpoint
from utils.dataset import get_dataset
from utils.embedding_loss import compute_embedding_loss
from utils.util import iou_fixed, get_lr_schedulers, show_image_summary, get_model

NUM_EPOCHS = 400
TRAIN_KITTI = False
MASK_CHANGE_THRESHOLD = 1000

BBOX_CROP = True
BEST_IOU=0

network_models = {0:"RGMP", 1:"FeatureAgg3d", 2: "FeatureAgg3dMergeTemporal", 3: "FeatureAgg3dMulti",
                  4: "FeatureAgg3dMulti101", 5: "Resnet3d", 6: "Resnet3dPredictOne", 7: "Resnet3dMaskGuidance",
                  8: "SiamResnet3d", 9:"Resnet3dNonLocal", 10: "Resnet3dSimilarity", 11:"Resnet3dEmbeddingNetwork",
                  12: "Resnet3dSegmentEmbedding"}
palette = Image.open(DAVIS_ROOT + '/Annotations/480p/bear/00000.png').getpalette()


def train(train_loader, model, criterion, optimizer, epoch, foo):
  global count
  batch_time = AverageMeter()
  data_time = AverageMeter()
  losses = AverageMeter()
  ious = AverageMeter()
  losses_extra = AverageMeter()

  # switch to train mode
  model.train()

  end = time.time()
  for i, input_dict in enumerate(train_loader):
    input, input_var, iou, loss, loss_image, masks_guidance, output, target, loss_extra = forward(criterion, input_dict, ious,
                                                                                      model)
    losses.update(loss.item(), input.size(0))
    losses_extra.update(loss_extra.item(), 1)
    foo.add_scalar("data/loss", loss, count)
    foo.add_scalar("data/iou", iou, count)
    if args.show_image_summary:
      if "proposals" in input_dict:
        foo.add_images("data/proposals", input_dict['proposals'][:, :, -1].repeat(1, 3, 1, 1), count)
      show_image_summary(count, foo, input_var[0:1], masks_guidance, target[0:1], output[0:1])

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()
    count = count + 1

    print('Epoch: [{0}][{1}/{2}]\t'
          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
          'Loss Extra {loss_extra.val:.4f}({loss_extra.avg:.4f})\t'
          'IOU {iou.val:.4f} ({iou.avg:.4f})\t'.format(
      epoch, i * args.bs, len(train_loader)*args.bs, batch_time=batch_time,
      data_time=data_time, loss=losses, iou=ious, loss_extra=losses_extra), flush=True)

  print('Finished Train Epoch {} Loss {losses.avg:.5f} Loss Extra {losses_extra.avg: .5f} IOU {iou.avg: .5f}'.format(epoch, losses=losses, losses_extra=losses_extra, iou=ious), flush=True)
  return losses.avg, ious.avg


def forward(criterion, input_dict, ious, model):
  input = input_dict["images"]
  target = input_dict["target"]
  target_extra = None if 'target_extra' not in input_dict else input_dict['target_extra']
  if 'masks_guidance' in input_dict:
    masks_guidance = input_dict["masks_guidance"]
    masks_guidance = masks_guidance.float().cuda()
  else:
    masks_guidance = None
  info = input_dict["info"]
  # data_time.update(time.time() - end)
  input_var = input.float().cuda()
  # compute output
  pred = run_forward(model, input_var, masks_guidance, input_dict['proposals'])
  loss, loss_image, pred, loss_extra = compute_loss(criterion, pred, target, target_extra)
  pred = F.softmax(pred, dim=1)
  iou = iou_fixed(pred.data.cpu().numpy(), target.data.cpu().numpy())
  ious.update(np.mean(iou))
  return input, input_var, iou, loss, loss_image, masks_guidance, pred, target, loss_extra


def compute_loss(criterion, pred, target, target_extra = None):
  pred_mask = pred[0]
  pred_extra = pred[1] if len(pred) > 1 else None
  if len(pred_mask.shape) > 4:
    pred_mask = F.interpolate(pred_mask, target.shape[2:], mode="trilinear")
  else:
    pred_mask = F.interpolate(pred_mask, target.shape[2:], mode="bilinear")

  if isinstance(criterion, torch.nn.CrossEntropyLoss):
    loss_image = criterion(pred_mask, target.squeeze(1).cuda().long())
  else:
    loss_image = criterion(pred_mask[:, -1], target.squeeze(1).cuda().float())
  loss_mask = bootstrapped_ce_loss(loss_image)

  loss_extra = 0
  if pred_extra is not None:
    # estimate loss for pixel level similarity
    if 'similarity' in args.losses:
      # get reference similarity mask
      assert 'similarity_ref' in target_extra
      batch_size = pred_extra.shape[0]
      similarity_ref = target_extra['similarity_ref'][:, :, 0].cuda().float()
      similarity_ref = F.interpolate(similarity_ref, scale_factor=[0.125,0.125], mode='nearest')

      # restore the time dimension
      pred_extra = F.interpolate(pred_extra.unsqueeze(1), scale_factor=[2, 2], mode='bilinear').squeeze(1)
      shape = similarity_ref.shape[2:]
      A = F.softmax(pred_extra.exp(), dim=-1)
      # A = A.contiguous()
      # A = A.view(tuple(pred_extra.shape[:2],)  + (-1,) + (shape[0]*shape[1],))


      similarity_ref = similarity_ref.unsqueeze(2).repeat(1, 1, 8, 1, 1)
      similarity_ref = similarity_ref.cuda().reshape(tuple(similarity_ref.shape[:2],) + (-1,)).\
        permute(0,2,1).float()

      y = torch.matmul(A, similarity_ref).permute(0,2,1)
      # interpolate and sample the similarity with the first frame instances
      original_size = (batch_size, target_extra['similarity_ref'].shape[1], target_extra['similarity_raw_mask'].shape[2] + 1,
      target_extra['similarity_raw_mask'].shape[-1], target_extra['similarity_raw_mask'].shape[-2],)
      y = y.reshape(tuple(y.shape[:2],) + (original_size[2],) + tuple(shape,)).contiguous()
      y = F.interpolate(y, size=original_size[2:], mode='trilinear')
      y = y[:, :, 1:]

      # y = y.view(tuple(y.shape[:3],)+ tuple(target_extra['similarity_raw_mask'].shape[-2:],))[:, :, 1:]

      # compute loss
      criterion_extra = torch.nn.CrossEntropyLoss(reduce=False)
      similarity_target = target_extra['similarity_raw_mask'].squeeze(1).cuda().long()
      loss_extra = criterion_extra(y, similarity_target)
      loss_extra = bootstrapped_ce_loss(loss_extra)
    if "embedding" in args.losses:
      pred_extra = F.interpolate(pred_extra, scale_factor=(1,8,8), mode='trilinear')
      loss_extra, _, _ = compute_embedding_loss(pred_extra, target_extra['similarity_ref'].cuda())

  # print("loss_extra {}".format(loss_extra))
  loss = loss_mask + loss_extra

  return loss, loss_image, pred_mask, loss_extra


def validate(val_loader, model, criterion, epoch, foo):
  batch_time = AverageMeter()
  losses = AverageMeter()
  ious = AverageMeter()
  losses_extra = AverageMeter()

  # switch to evaluate mode
  model.eval()

  end = time.time()
  print("Starting validation for epoch {}".format(epoch), flush=True)
  for seq in val_loader.dataset.get_video_ids():
    val_loader.dataset.set_video_id(seq)
    ious_video = AverageMeter()
    for i, input_dict in enumerate(val_loader):
      with torch.no_grad():
        # compute output
        input, input_var, iou, loss, loss_image, masks_guidance, output, target, loss_extra = forward(criterion, input_dict, ious,
                                                                                                      model)
        ious_video.update(np.mean(iou))
        losses.update(loss.item(), input.size(0))
        losses_extra.update(loss_extra.item(), 1)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.show_image_summary:
          show_image_summary(count, foo, input_var, masks_guidance, target, output)

        print('Test: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.5f})\t'
              'Loss Extra {loss_extra.val:.4f} ({loss_extra.avg:.5f})\t'
              'IOU {iou.val:.4f} ({iou.avg:.5f})\t'.format(
          input_dict['info']['name'], i, len(val_loader), batch_time=batch_time, loss=losses, iou=ious_video,
          loss_extra=losses_extra),
          flush=True)
    print('Sequence {0}\t IOU {iou.avg}'.format(input_dict['info']['name'], iou=ious_video), flush=True)
    ious.update(ious_video.avg)

    foo.add_scalar("data/losses-test", losses.avg, epoch)

  print('Finished Eval Epoch {} Loss {losses.avg:.5f} Losses Extra {losses_extra.avg} IOU {iou.avg: 5f}'
        .format(epoch, losses=losses, losses_extra=losses_extra, iou=ious), flush=True)

  return losses.avg, ious.avg


if __name__ == '__main__':
    args = parse_args()
    count = 0
    MODEL_DIR = os.path.join('saved_models', args.network_name)
    print("Arguments used: {}".format(args), flush=True)

    trainset, testset = get_dataset(args)
    # trainset = DAVIS(DAVIS_ROOT, imset='2017/train.txt', is_train=True,
    #                  random_instance=args.random_instance, crop_size=args.crop_size, resize_mode=args.resize_mode)
    # trainset = YoutubeVOSDataset(YOUTUBEVOS_ROOT, imset='train', is_train=True,
    #                  random_instance=args.random_instance, crop_size=args.crop_size, resize_mode=args.resize_mode)
    # testset = DAVISEval(DAVIS_ROOT, imset='2017/val.txt', random_instance=False, crop_size=args.crop_size_eval,
    #                     resize_mode=args.resize_mode_eval)
    # sample a subset of data for testing
    sampler = RandomSampler(trainset, replacement=True, num_samples=args.data_sample) \
      if args.data_sample is not None else None
    shuffle = True if args.data_sample is None else False
    trainloader = DataLoader(trainset, batch_size=args.bs, num_workers=args.num_workers, shuffle=shuffle, sampler=sampler)
    testloader = DataLoader(testset, batch_size=1, num_workers=1, shuffle=False)
    model = get_model(args, network_models)

    # model = FeatureAgg3dMergeTemporal()
    print("Using model: {}".format(model.__class__), flush=True)
    print(args)

    if torch.cuda.is_available():
      device_ids = [0, 1]
      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

      print("devices available: {}".format(torch.cuda.device_count()))
      model = torch.nn.DataParallel(model)
      model.cuda()

    # model.cuda()
    # print(summary(model, tuple((256,256)), batch_size=1))
    writer = SummaryWriter(log_dir="runs/" + args.network_name)
    optimizer = torch.optim.Adam(model.module.parameters(), lr=args.lr)
    model, optimizer, start_epoch, best_iou_train, best_iou_eval, best_loss_train, best_loss_eval = \
      load_weights(model, optimizer, args, MODEL_DIR, scheduler=None)# params
    lr_schedulers = get_lr_schedulers(optimizer, args, start_epoch)

    params = []
    for key, value in dict(model.named_parameters()).items():
      if value.requires_grad:
        params += [{'params':[value],'lr':args.lr, 'weight_decay': 4e-5}]

    criterion = torch.nn.BCEWithLogitsLoss(reduce=False) if args.n_classes == 2 else \
      torch.nn.CrossEntropyLoss(reduce=False)
    print("Using {} criterion", criterion)
    # iters_per_epoch = len(Trainloader)
    # model.eval()

    if args.task == "train":
      # best_loss = best_loss_train
      # best_iou = best_iou_train
      if args.freeze_bn:
        encoders = [module for module in model.modules() if isinstance(module, Encoder)]
        for encoder in encoders:
          encoder.freeze_batchnorm()
      for epoch in range(start_epoch, args.num_epochs):
        loss_mean, iou_mean  = train(trainloader, model, criterion, optimizer, epoch, writer)
        for lr_scheduler in lr_schedulers:
          lr_scheduler.step(epoch)
        if iou_mean > best_iou_train or loss_mean < best_loss_train:
          if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
          best_iou_train = iou_mean if iou_mean > best_iou_train else best_iou_train
          best_loss_train = loss_mean if loss_mean < best_loss_train else best_loss_train
          save_name = '{}/{}.pth'.format(MODEL_DIR, "model_best_train")
          save_checkpoint(epoch, iou_mean, loss_mean, model, optimizer, save_name, is_train=True,
                          scheduler=None)

        if (epoch + 1) % args.eval_epoch == 0:
          loss_mean, iou_mean = validate(testloader, model, criterion, epoch, writer)
          if iou_mean > best_iou_eval:
            best_iou_eval = iou_mean
            save_name = '{}/{}.pth'.format(MODEL_DIR, "model_best_eval")
            save_checkpoint(epoch, iou_mean, loss_mean, model, optimizer, save_name, is_train=False,
                            scheduler=lr_scheduler)
    elif args.task == 'eval':
      validate(testloader, model, criterion, 1, writer)
    elif 'infer' in args.task:
      infer(args, testloader, model, criterion, writer)
    else:
      raise ValueError("Unknown task {}".format(args.task))

