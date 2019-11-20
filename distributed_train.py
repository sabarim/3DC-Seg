import os
import time

import apex
import numpy as np
import torch
from PIL import Image
from apex import amp
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler
from torchsummary import summary

from Forward import forward
from inference_handlers.inference import infer
from network.RGMP import Encoder
from utils.Argparser import parse_args
# Constants
from utils.AverageMeter import AverageMeter
from utils.Constants import DAVIS_ROOT, network_models
from utils.Saver import load_weights, save_checkpoint
from utils.dataset import get_dataset
from utils.util import get_lr_schedulers, show_image_summary, get_model, init_torch_distributed, cleanup_env, \
  reduce_tensor

NUM_EPOCHS = 400
TRAIN_KITTI = False
MASK_CHANGE_THRESHOLD = 1000

BBOX_CROP = True
BEST_IOU=0
torch.backends.cudnn.benchmark=True


def train(train_loader, model, criterion, optimizer, epoch, foo):
  global count
  batch_time = AverageMeter()
  data_time = AverageMeter()
  losses = AverageMeter()
  ious = AverageMeter()
  losses_extra = AverageMeter()
  ious_extra = AverageMeter()

  # switch to train mode
  model.train()

  end = time.time()
  for i, input_dict in enumerate(train_loader):
    iou, loss, loss_image, output, loss_extra = \
      forward(args, criterion, input_dict, model, ious_extra=ious_extra)

    # compute gradient and do SGD step
    optimizer.zero_grad()
    with amp.scale_loss(loss, optimizer) as scaled_loss:
      scaled_loss.backward()
    #loss.backward()
    optimizer.step()

    # losses.update(loss.item(), 1)
    # losses_extra.update(loss_extra.item(), 1)

    # measure elapsed time
    # if i % args.print_freq == 0:
    count = count + 1

    # Average loss and accuracy across processes for logging
    if torch.cuda.device_count() > 1:
      reduced_loss = reduce_tensor(loss.data, args)
      reduced_loss_extra = reduce_tensor(loss_extra.data, args)
      reduced_iou = reduce_tensor(iou, args)
    else:
      reduced_loss = loss.data
      reduced_iou = iou.data
      reduced_loss_extra = loss_extra.data

    # to_python_float incurs a host<->device sync
    # FIXME: may device count is not the right parameter to use here
    losses.update(reduced_loss.item(), args.world_size)
    losses_extra.update(reduced_loss_extra.item(), args.world_size)
    ious.update(reduced_iou.item(), args.world_size)

    foo.add_scalar("data/loss", losses.val, count)
    foo.add_scalar("data/iou", ious.val, count)
    if args.show_image_summary:
      if "proposals" in input_dict:
        foo.add_images("data/proposals", input_dict['proposals'][:, :, -1].repeat(1, 3, 1, 1), count)
      masks_guidance = input_dict['masks_guidance'] if 'masks_guidance' in input_dict else None
      show_image_summary(count, foo, input_dict['images'][0:1].float(), masks_guidance,
                         input_dict['target'][0:1], output[0:1])

    torch.cuda.synchronize()
    batch_time.update((time.time() - end) / args.print_freq)
    end = time.time()

    if args.local_rank == 0:
      print('Epoch: [{0}][{1}/{2}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Loss Extra {loss_extra.val:.4f}({loss_extra.avg:.4f})\t'
            'IOU {iou.val:.4f} ({iou.avg:.4f})\t'
            'IOU Extra {iou_extra.val:.4f} ({iou_extra.avg:.4f})\t'.format(
        epoch, i*args.world_size*args.bs, len(trainloader) * args.bs*args.world_size,
               args.world_size * args.bs / batch_time.val,
               args.world_size * args.bs / batch_time.avg,
        batch_time=batch_time, loss=losses, iou=ious,
        loss_extra=losses_extra, iou_extra=ious_extra), flush=True)

  if args.local_rank == 0:
    print('Finished Train Epoch {} Loss {losses.avg:.5f} Loss Extra {losses_extra.avg: .5f} IOU {iou.avg: .2f} '
          'IOU Extra {iou_extra.avg: .2f}'.
          format(epoch, losses=losses, losses_extra=losses_extra, iou=ious, iou_extra=ious_extra), flush=True)
  return losses.avg, ious.avg


def validate(dataset, model, criterion, epoch, foo):
  global count
  count=0
  batch_time = AverageMeter()
  losses = AverageMeter()
  losses_extra = AverageMeter()
  ious = AverageMeter()
  ious_extra = AverageMeter()

  # switch to evaluate mode
  model.eval()

  end = time.time()
  print("Starting validation for epoch {}".format(epoch), flush=True)
  for seq in dataset.get_video_ids():
    dataset.set_video_id(seq)
    test_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
  # test_sampler.set_epoch(epoch)
    testloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False, sampler=test_sampler, pin_memory=True)
    ious_video = AverageMeter()
    ious_video_extra = AverageMeter()
    for i, input_dict in enumerate(testloader):
      with torch.no_grad():
        # compute output
        iou, loss, loss_image, output, loss_extra = forward(args, criterion, input_dict,
                                                            model, ious_extra=ious_video_extra)
        count = count + 1

        # Average loss and accuracy across processes for logging
        if torch.cuda.device_count() > 1:
          reduced_loss = reduce_tensor(loss.data, args)
          reduced_loss_extra = reduce_tensor(loss_extra.data, args)
          reduced_iou = reduce_tensor(iou, args)
        else:
          reduced_loss = loss.data
          reduced_loss_extra = loss_extra.data
          reduced_iou = iou.data

        # to_python_float incurs a host<->device sync
        # FIXME: may device count is not the right parameter to use here
        losses.update(reduced_loss.item(), args.world_size)
        losses_extra.update(reduced_loss_extra.item(), args.world_size)
        ious_video.update(reduced_iou.item(), args.world_size)

        foo.add_scalar("data/loss", losses.val, count)
        foo.add_scalar("data/iou", ious.val, count)

        if args.show_image_summary:
          masks_guidance = input_dict['masks_guidance'] if 'masks_guidance' in input_dict else None
          show_image_summary(count, foo, input_dict['images'], masks_guidance, input_dict['target'],
                             output)

        torch.cuda.synchronize()
        batch_time.update((time.time() - end) / args.print_freq)
        end = time.time()

        if args.local_rank == 0:
          print('Test: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.5f})\t'
                'Loss Extra {loss_extra.val:.4f} ({loss_extra.avg:.5f})\t'
                'IOU {iou.val:.4f} ({iou.avg:.5f})\t'
                'IOU Extra {iou_extra.val:.4f} ({iou_extra.avg:.5f})\t'.format(
            input_dict['info']['name'], i*args.world_size, len(testloader)*args.world_size,
                                        args.world_size * args.bs / batch_time.val,
                                        args.world_size * args.bs / batch_time.avg,
            batch_time=batch_time, loss=losses, iou=ious_video,
            loss_extra=losses_extra, iou_extra=ious_extra),
            flush=True)
    if args.local_rank == 0:
      print('Sequence {0}\t IOU {iou.avg} IOU Extra {iou_extra.avg}'.format(input_dict['info']['name'], iou=ious_video,
                                                                            iou_extra = ious_video_extra), flush=True)
    ious.update(ious_video.avg)
    ious_extra.update(ious_video_extra.avg)

  foo.add_scalar("data/losses-test", losses.avg, epoch)

  if args.local_rank == 0:
    print('Finished Eval Epoch {} Loss {losses.avg:.5f} Losses Extra {losses_extra.avg} IOU {iou.avg: .2f} '
          'IOU Extra {iou_extra.avg: .2f}'
          .format(epoch, losses=losses, losses_extra=losses_extra, iou=ious, iou_extra = ious_extra), flush=True)

  return losses.avg, ious.avg


if __name__ == '__main__':
    args = parse_args()
    count = 0
    local_rank = 0
    device = None
    MODEL_DIR = os.path.join('saved_models', args.network_name)
    writer = SummaryWriter(log_dir="runs/" + args.network_name)
    print("Arguments used: {}".format(args), flush=True)

    trainset, testset = get_dataset(args)
    #train_sampler = RandomSampler(trainset, replacement=True, num_samples=args.data_sample) \
    #  if args.data_sample is not None else \
    #  torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True)
    #shuffle = True if args.data_sample is None else False
    model = get_model(args, network_models)

    # model = FeatureAgg3dMergeTemporal()
    args.world_size = 1
    print("Using model: {}".format(model.__class__), flush=True)
    print(args)

    if torch.cuda.is_available():
      torch.cuda.set_device(args.local_rank)
      init_torch_distributed()
      model = apex.parallel.convert_syncbn_model(model)
      model.cuda()
      optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
      model, optimizer, start_epoch, best_iou_train, best_iou_eval, best_loss_train, best_loss_eval, amp_weights = \
        load_weights(model, optimizer, args, MODEL_DIR, scheduler=None, amp=amp)  # params
      lr_schedulers = get_lr_schedulers(optimizer, args, start_epoch)

      opt_levels = {'fp32': 'O0', 'fp16': 'O2', 'mixed':'O1'}
      if args.precision in opt_levels:
        opt_level = opt_levels[args.precision]
      else:
        print('WARN: Precision string is not understood. Falling back to fp32')
      print("opt_level is {}".format(opt_level))
      model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
      # amp.load_state_dict(amp_weights)
      model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)
      args.world_size = torch.distributed.get_world_size()

    train_sampler = torch.utils.data.distributed.DistributedSampler(
      torch.utils.data.RandomSampler(trainset, replacement=True, num_samples=args.data_sample),
      shuffle=True)
    shuffle = True if args.data_sample is None else False

    trainloader = DataLoader(trainset, batch_size=args.bs, num_workers=args.num_workers,
                             sampler=train_sampler)

    # print(summary(model, tuple((256,256)), batch_size=1))

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
        train_sampler.set_epoch(epoch)
        loss_mean, iou_mean  = train(trainloader, model, criterion, optimizer, epoch, writer)
        for lr_scheduler in lr_schedulers:
          lr_scheduler.step(epoch)

        if args.local_rank == 0:
          if iou_mean > best_iou_train or loss_mean < best_loss_train:
            if not os.path.exists(MODEL_DIR):
              os.makedirs(MODEL_DIR)
            best_iou_train = iou_mean if iou_mean > best_iou_train else best_iou_train
            best_loss_train = loss_mean if loss_mean < best_loss_train else best_loss_train
            save_name = '{}/{}.pth'.format(MODEL_DIR, "model_best_train")
            save_checkpoint(epoch, iou_mean, loss_mean, model, optimizer, save_name, is_train=True,
                            scheduler=None, amp=amp)

        if (epoch + 1) % args.eval_epoch == 0:
          loss_mean, iou_mean = validate(testset, model, criterion, epoch, writer)
          if iou_mean > best_iou_eval and args.local_rank == 0:
            best_iou_eval = iou_mean
            save_name = '{}/{}.pth'.format(MODEL_DIR, "model_best_eval")
            save_checkpoint(epoch, iou_mean, loss_mean, model, optimizer, save_name, is_train=False,
                            scheduler=lr_scheduler)
    elif args.task == 'eval':
      validate(testset, model, criterion, 1, writer)
    elif 'infer' in args.task:
      infer(args, testset, model, criterion, writer, distributed=True)
    else:
      raise ValueError("Unknown task {}".format(args.task))

    cleanup_env()

