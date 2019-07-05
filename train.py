import importlib
import inspect
import os
import time

import numpy as np
import torch
from PIL import Image
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler

from datasets.DAVIS import DAVIS
from network.FeatureAgg3d import FeatureAgg3d
from network.models import BaseNetwork
from utils.Argparser import parse_args
# Constants
from utils.AverageMeter import AverageMeter
from utils.Constants import DAVIS_ROOT
from utils.Loss import bootstrapped_ce_loss
from utils.Saver import load_weights, save_checkpoint
from utils.util import iou_fixed, all_subclasses

NUM_EPOCHS = 400
TRAIN_KITTI = False
MASK_CHANGE_THRESHOLD = 1000

BBOX_CROP = True
RANDOM_INSTANCE = True
BEST_IOU=0

network_models = {0:"RGMP", 1:"FeatureAgg3d"}
palette = Image.open(DAVIS_ROOT + '/Annotations/480p/bear/00000.png').getpalette()


def propagate(model, inputs, ref_mask):
  refs = []
  assert inputs.shape[2] >= 2
  for i in range(inputs.shape[2]):
    r5, r4, r3, r2 = model.encoder(inputs[:, :, i], ref_mask[:, :, i])
    refs+=[r5.unsqueeze(2)]
  support = torch.cat(refs[:-1], dim=2)
  e2 = model.decoder(r5, r4, r3, r2, support)

  return (F.softmax(e2[0], dim=1), r5, e2[-1])


def train(train_loader, model, criterion, optimizer, epoch, foo):
  global count
  batch_time = AverageMeter()
  data_time = AverageMeter()
  losses = AverageMeter()
  ious = AverageMeter()

  # switch to train mode
  model.train()

  end = time.time()
  for i, input_dict in enumerate(train_loader):
    input, input_var, iou, loss, loss_image, masks_guidance, output, target = forward(criterion, input_dict, ious,
                                                                                      model)
    losses.update(loss.item(), input.size(0))
    foo.add_scalar("data/loss", loss, count)
    foo.add_scalar("data/iou", iou, count)
    if args.show_image_summary:
      foo.add_image("data/input", input_var[:, :3], count)
      foo.add_image("data/guidance", masks_guidance, count)
      foo.add_image("data/loss_image", loss_image.unsqueeze(1), count)
      foo.add_image("data/target", target, count)
      foo.add_image("data/pred", output.unsqueeze(1), count)

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
          'IOU {iou.val:.4f} ({iou.avg:.4f})\t'.format(
      epoch, i * args.bs, len(train_loader)*args.bs, batch_time=batch_time,
      data_time=data_time, loss=losses, iou=ious))

  print('Finished Train Epoch {} Loss {losses.avg:.5f} IOU {iou.avg: 5f}'
        .format(epoch, losses=losses, iou=ious))
  return losses.avg, ious.avg


def forward(criterion, input_dict, ious, model):
  input = input_dict["images"]
  target = input_dict["target"]
  masks_guidance = input_dict["masks_guidance"]
  info = input_dict["info"]
  # data_time.update(time.time() - end)
  masks_guidance = masks_guidance.float().cuda()
  input_var = input.float().cuda()
  # compute output
  pred = propagate(model, input_var, masks_guidance)
  pred = F.interpolate(pred[0], target.shape[2:], mode="bilinear")
  # output = F.sigmoid(pred[:, -1])
  # output = F.softmax(pred, dim=1)
  loss_image = criterion(pred[:, -1], target.squeeze(1).cuda().float())
  loss = bootstrapped_ce_loss(loss_image)
  # loss = criterion(output, target.squeeze(1)).mean()
  iou = iou_fixed(pred.data.cpu().numpy(), target.data.cpu().numpy())
  ious.update(np.mean(iou))
  return input, input_var, iou, loss, loss_image, masks_guidance, pred, target


def validate(val_loader, model, criterion, epoch, foo):
  batch_time = AverageMeter()
  losses = AverageMeter()
  ious = AverageMeter()

  # switch to evaluate mode
  model.eval()

  end = time.time()
  print("Starting validation for epoch {}".format(epoch))
  for i, input_dict in enumerate(val_loader):
    with torch.no_grad():
      # compute output
      input, input_var, iou, loss, loss_image, masks_guidance, output, target = forward(criterion, input_dict, ious,
                                                                                        model)
      loss_image = criterion(output, target.squeeze(1))
      loss = bootstrapped_ce_loss(loss_image)
      ious.update(np.mean(iou))
      losses.update(loss.item(), input.size(0))
      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()

      print('Test: [{0}/{1}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.5f})\t'
            'IOU {iou.val:.4f} ({iou.avg:.5f})\t'.format(
        i, len(val_loader), batch_time=batch_time, loss=losses, iou=ious))

  foo.add_scalar("data/losses-test", losses.avg, epoch)

  print('Finished Eval Epoch {} Loss {losses.avg:.5f} IOU {iou.avg: 5f}'
        .format(epoch, losses=losses, iou=ious))

  return losses.avg, ious.avg


if __name__ == '__main__':
    args = parse_args()
    count = 0
    MODEL_DIR = os.path.join('saved_models', args.network_name)
    print("Arguments used: {}".format(args))

    trainset = DAVIS(DAVIS_ROOT, imset='2017/train.txt', is_train=True,
                     random_instance=RANDOM_INSTANCE, crop_size=(256, 256))
    testset = DAVIS(DAVIS_ROOT, imset='2017/val.txt')
    # sample a subset of data for testing
    sampler = RandomSampler(trainset, replacement=True, num_samples=args.data_sample) \
      if args.data_sample is not None else None
    shuffle = True if args.data_sample is None else False
    trainloader = DataLoader(trainset, batch_size=args.bs, num_workers=args.num_workers, shuffle=shuffle, sampler=sampler)
    testloader = DataLoader(testset, batch_size=1, num_workers=args.num_workers)

    model_classes = all_subclasses(BaseNetwork)
    class_index = [cls.__name__ for cls in model_classes].index(network_models[args.network])
    model = list(model_classes)[class_index]()
    model = FeatureAgg3d()
    print("Using model: {}".format(model.__class__))
    print(args)

    if torch.cuda.is_available():
        model.cuda()

    # print(summary(model, tuple((256,256)), batch_size=1))
    writer = SummaryWriter(log_dir="runs/" + MODEL_DIR)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model, optimizer, start_epoch = load_weights(model, optimizer, args.loadepoch, MODEL_DIR)# params

    params = []
    for key, value in dict(model.named_parameters()).items():
      if value.requires_grad:
        params += [{'params':[value],'lr':args.lr, 'weight_decay': 4e-5}]

    criterion = torch.nn.BCELoss(reduce=False)
    # iters_per_epoch = len(Trainloader)
    model.eval()
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95) if args.adaptive_lr else None

    if args.task == "train":
      best_loss = 0
      best_iou = 0
      best_loss_eval = 0
      best_iou_eval = 0
      if args.freeze_bn:
        model.encoder.freeze_batchnorm()
      for epoch in range(start_epoch, args.num_epochs):
        loss_mean, iou_mean  = train(trainloader, model, criterion, optimizer, epoch, writer)
        if lr_scheduler is not None:
          lr_scheduler.step(epoch)
        if iou_mean > best_iou or loss_mean < best_loss:
          if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
          save_name = '{}/{}.pth'.format(MODEL_DIR, "model_best_train")
          save_checkpoint(epoch, iou_mean, model, optimizer, save_name)

        if (epoch + 1) % args.eval_epoch == 0:
          loss_mean, iou_mean = validate(testloader, model, criterion, epoch, writer)
          if iou_mean > best_iou_eval:
            save_name = '{}/{}.pth'.format(MODEL_DIR, "model_best_eval")
            save_checkpoint(epoch, iou_mean, model, optimizer, save_name)
    else:
      validate(testloader, model, criterion, 1, writer)

