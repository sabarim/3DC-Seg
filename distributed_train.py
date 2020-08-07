import os
import time
import signal
import apex
import torch
from apex import amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from Forward import forward
from inference_handlers.inference import infer
from network.RGMP import Encoder
from utils.Argparser import parse_args
# Constants
from utils.AverageMeter import AverageMeter, AverageMeterDict
from utils.Constants import network_models
from utils.Saver import load_weights, save_checkpoint
from utils.dataset import get_dataset
from utils.util import show_image_summary, get_model_from_args, init_torch_distributed, cleanup_env, \
    reduce_tensor, is_main_process, synchronize, get_lr_schedulers_args

NUM_EPOCHS = 400
TRAIN_KITTI = False
MASK_CHANGE_THRESHOLD = 1000

BBOX_CROP = True
BEST_IOU=0
torch.backends.cudnn.benchmark=True


class Trainer:
    def __init__(self, args):
        self.model_dir = os.path.join('saved_models', args.network_name)
        self.writer = SummaryWriter(log_dir="runs/" + args.network_name)
        print("Arguments used: {}".format(args), flush=True)
        self.trainset, self.testset = get_dataset(args)
        self.model = get_model_from_args(args, network_models)
        args.world_size = 1
        print("Using model: {}".format(self.model.__class__), flush=True)
        print(args)
        self.args = args
        self.iteration = 0
        self.epoch = 0
        self.best_iou_train = 0
        self.best_iou_eval = 0
        self.best_loss_train = 0
        self.losses = AverageMeter()
        self.ious = AverageMeter()
        self.losses_extra = AverageMeterDict()
        self.ious_extra = AverageMeter()

        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.model, self.optimiser, self.lr_schedulers, self.epoch, \
            self.best_iou_train, self.best_iou_eval = self.init_distributed(args)
        # TODO: do not use distributed package in this case
        elif torch.cuda.is_available():
            self.model, self.optimiser, self.lr_schedulers, self.epoch, \
            self.best_iou_train, self.best_iou_eval = self.init_distributed(args)
        else:
            raise RuntimeError("CUDA not available.")
        # shuffle parameter does not seem to shuffle the data for distributed sampler
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(
            torch.utils.data.RandomSampler(self.trainset, replacement=True, num_samples=args.data_sample),
            shuffle=True)
        shuffle = True if self.train_sampler is None else False
        self.trainloader = DataLoader(self.trainset, batch_size=args.bs, num_workers=args.num_workers,
                                 shuffle=shuffle, sampler=self.train_sampler)

        print(summary(self.model, tuple((3,args.tw,256,256)), batch_size=1))
        params = []
        for key, value in dict(self.model.named_parameters()).items():
            if value.requires_grad:
                params += [{'params': [value], 'lr': args.lr, 'weight_decay': 4e-5}]
        self.criterion = torch.nn.CrossEntropyLoss(reduce=False) if 'cce' in args.losses else \
            torch.nn.BCEWithLogitsLoss(reduce=False)
        print("Using {} criterion", self.criterion)

    def init_distributed(self, args):
        torch.cuda.set_device(args.local_rank)
        init_torch_distributed()
        model = apex.parallel.convert_syncbn_model(self.model)
        model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        model, optimizer, start_epoch, best_iou_train, best_iou_eval, best_loss_train, best_loss_eval, amp_weights = \
            load_weights(model, optimizer, args, self.model_dir, scheduler=None, amp=amp)  # params
        lr_schedulers = get_lr_schedulers_args(optimizer, args, start_epoch)
        opt_levels = {'fp32': 'O0', 'fp16': 'O2', 'mixed': 'O1'}
        if args.precision in opt_levels:
            opt_level = opt_levels[args.precision]
        else:
            print('WARN: Precision string is not understood. Falling back to fp32')
        print("opt_level is {}".format(opt_level))
        model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
        # amp.load_state_dict(amp_weights)
        model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)
        args.world_size = torch.distributed.get_world_size()
        return model, optimizer, lr_schedulers, start_epoch, best_iou_train, best_iou_eval

    def train(self):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        # switch to train mode
        self.model.train()
        self.ious.reset()
        self.ious_extra.reset()
        self.losses.reset()
        self.losses_extra.reset()

        end = time.time()
        for i, input_dict in enumerate(self.trainloader):
            iou, loss, loss_image, output, loss_extra = \
                forward(args, self.criterion, input_dict, self.model, ious_extra=self.ious_extra, summarywriter=self.writer)

            # compute gradient and do SGD step
            self.optimiser.zero_grad()
            with amp.scale_loss(loss, self.optimiser) as scaled_loss:
                scaled_loss.backward()
            # loss.backward()
            self.optimiser.step()
            self.iteration += 1

            # Average loss and accuracy across processes for logging
            if torch.cuda.device_count() > 1:
                reduced_loss = reduce_tensor(loss, args).data.item()
                reduced_loss_extra = dict(
                    [(key, reduce_tensor(val, args).data.item()) for key, val in loss_extra.items()])
                reduced_iou = reduce_tensor(iou, args).data.item()
            else:
                reduced_loss = loss.data.item()
                reduced_loss_extra = dict([(key, val.data.item()) for key, val in loss_extra.items()])
                reduced_iou = iou.data.item()
                # reduced_loss_extra = loss_extra.data

            # to_python_float incurs a host<->device sync
            # FIXME: may device count is not the right parameter to use here
            self.losses.update(reduced_loss, args.world_size)
            self.losses_extra.update(reduced_loss_extra, args.world_size)
            self.ious.update(reduced_iou, args.world_size)

            self.writer.add_scalar("data/loss", self.losses.val, self.iteration)
            self.writer.add_scalar("data/iou", self.ious.val, self.iteration)
            if args.show_image_summary:
                if "proposals" in input_dict:
                    self.writer.add_images("data/proposals", input_dict['proposals'][:, :, -1].repeat(1, 3, 1, 1),
                                           self.iteration)
                masks_guidance = input_dict['masks_guidance'] if 'masks_guidance' in input_dict else None
                show_image_summary(self.iteration, self.writer, input_dict['images'][0:1].float(), masks_guidance,
                                   input_dict['target'][0:1], output[0:1])

            torch.cuda.synchronize()
            batch_time.update((time.time() - end) / args.print_freq)
            end = time.time()

            if args.local_rank == 0:
                print('[Iter: {0}]Epoch: [{1}][{2}/{3}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Loss Extra {loss_extra}\t'
                      'IOU {iou.val:.4f} ({iou.avg:.4f})\t'
                      'IOU Extra {iou_extra.val:.4f} ({iou_extra.avg:.4f})\t'.format(
                    self.iteration, self.epoch, i * args.world_size * args.bs, len(self.trainloader) * args.bs * args.world_size,
                           args.world_size * args.bs / batch_time.val,
                           args.world_size * args.bs / batch_time.avg,
                    batch_time=batch_time, loss=self.losses, iou=self.ious,
                    loss_extra=self.losses_extra, iou_extra=self.ious_extra), flush=True)

        if args.local_rank == 0:
            print('Finished Train Epoch {} Loss {losses.avg:.5f} Loss Extra {losses_extra.avg} IOU {iou.avg: .2f} '
                  'IOU Extra {iou_extra.avg: .2f}'.
                  format(self.epoch, losses=self.losses, losses_extra=self.losses_extra, iou=self.ious,
                         iou_extra=self.ious_extra), flush=True)

        return self.losses.avg, self.ious.avg

    def eval(self):
        batch_time = AverageMeter()
        losses = AverageMeter()
        losses_extra = AverageMeterDict()
        ious = AverageMeter()
        ious_extra = AverageMeter()
        count=0
        # switch to evaluate mode
        self.model.eval()

        end = time.time()
        print("Starting validation for epoch {}".format(self.epoch), flush=True)
        for seq in self.testset.get_video_ids():
            self.testset.set_video_id(seq)
            test_sampler = torch.utils.data.distributed.DistributedSampler(self.testset, shuffle=False)
            # test_sampler.set_epoch(epoch)
            testloader = DataLoader(self.testset, batch_size=1, num_workers=1, shuffle=False, sampler=test_sampler,
                                    pin_memory=True)
            ious_video = AverageMeter()
            ious_video_extra = AverageMeter()
            for i, input_dict in enumerate(testloader):
                with torch.no_grad():
                    # compute output
                    iou, loss, loss_image, output, loss_extra = forward(args, self.criterion, input_dict,
                                                                        self.model, ious_extra=ious_video_extra)
                    count = count + 1

                    # Average loss and accuracy across processes for logging
                    if torch.cuda.device_count() > 1:
                        reduced_loss = reduce_tensor(loss.data, args)
                        reduced_loss_extra = dict(
                            [(key, reduce_tensor(val, args).data.item()) for key, val in loss_extra.items()])
                        reduced_iou = reduce_tensor(iou, args)
                    else:
                        reduced_loss = loss.data
                        reduced_loss_extra = dict([(key, val) for key, val in loss_extra.items()])
                        reduced_iou = iou.data

                    # to_python_float incurs a host<->device sync
                    # FIXME: may device count is not the right parameter to use here
                    losses.update(reduced_loss.item(), args.world_size)
                    losses_extra.update(reduced_loss_extra, args.world_size)
                    ious_video.update(reduced_iou.item(), args.world_size)

                    self.writer.add_scalar("data/loss", losses.val, count)
                    self.writer.add_scalar("data/iou", ious.val, count)

                    if args.show_image_summary:
                        masks_guidance = input_dict['masks_guidance'] if 'masks_guidance' in input_dict else None
                        show_image_summary(count, self.writer, input_dict['images'], masks_guidance, input_dict['target'],
                                           output)

                    torch.cuda.synchronize()
                    batch_time.update((time.time() - end) / args.print_freq)
                    end = time.time()

                    if args.local_rank == 0:
                        print('Test: [{0}][{1}/{2}]\t'
                              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                              'Loss {loss.val:.4f} ({loss.avg:.5f})\t'
                              'Loss Extra {loss_extra}\t'
                              'IOU {iou.val:.4f} ({iou.avg:.5f})\t'
                              'IOU Extra {iou_extra.val:.4f} ({iou_extra.avg:.5f})\t'.format(
                            input_dict['info']['name'], i * args.world_size, len(testloader) * args.world_size,
                                                        args.world_size * args.bs / batch_time.val,
                                                        args.world_size * args.bs / batch_time.avg,
                            batch_time=batch_time, loss=losses, iou=ious_video,
                            loss_extra=losses_extra, iou_extra=ious_video_extra),
                            flush=True)
            if args.local_rank == 0:
                print('Sequence {0}\t IOU {iou.avg} IOU Extra {iou_extra.avg}'.format(input_dict['info']['name'],
                                                                                      iou=ious_video,
                                                                                      iou_extra=ious_video_extra),
                      flush=True)
            ious.update(ious_video.avg)
            ious_extra.update(ious_video_extra.avg)

        self.writer.add_scalar("data/losses-test", losses.avg, self.epoch)

        if args.local_rank == 0:
            print('Finished Eval Epoch {} Loss {losses.avg:.5f} Losses Extra {losses_extra.avg} IOU {iou.avg: .2f} '
                  'IOU Extra {iou_extra.avg: .2f}'
                  .format(self.epoch, losses=losses, losses_extra=losses_extra, iou=ious, iou_extra=ious_extra), flush=True)

        return losses.avg, ious.avg

    def start(self):
        if args.task == "train":
            # best_loss = best_loss_train
            # best_iou = best_iou_train
            if args.freeze_bn:
                encoders = [module for module in self.model.modules() if isinstance(module, Encoder)]
                for encoder in encoders:
                    encoder.freeze_batchnorm()
            
            start_epoch = self.epoch
            for epoch in range(start_epoch, args.num_epochs):
                self.epoch = epoch
                if self.train_sampler is not None:
                    self.train_sampler.set_epoch(epoch)
                loss_mean, iou_mean = self.train()
                for lr_scheduler in self.lr_schedulers:
                    lr_scheduler.step(epoch)

                if args.local_rank == 0:
                    if iou_mean > self.best_iou_train or loss_mean < self.best_loss_train:
                        if not os.path.exists(self.model_dir):
                            os.makedirs(self.model_dir)
                        self.best_iou_train = iou_mean if iou_mean > self.best_iou_train else self.best_iou_train
                        self.best_loss_train = loss_mean if loss_mean < self.best_loss_train else self.best_loss_train
                        save_name = '{}/{}.pth'.format(self.model_dir, "model_best_train")
                        save_checkpoint(epoch, iou_mean, loss_mean, self.model, self.optimiser, save_name, is_train=True,
                                        scheduler=None, amp=amp)

                if (epoch + 1) % args.eval_epoch == 0:
                    loss_mean, iou_mean = self.eval()
                    if iou_mean > self.best_iou_eval and args.local_rank == 0:
                        self.best_iou_eval = iou_mean
                        save_name = '{}/{}.pth'.format(self.model_dir, "model_best_eval")
                        save_checkpoint(epoch, iou_mean, loss_mean, self.model, self.optimiser, save_name, is_train=False,
                                        scheduler=self.lr_schedulers)
        elif args.task == 'eval':
            self.eval()
        elif 'infer' in args.task:
            infer(args, self.testset, self.model, self.criterion, self.writer, distributed=True)
        else:
            raise ValueError("Unknown task {}".format(args.task))

    def backup_session(self, signalNumber, _):
        if is_main_process() and self.args.task == 'train':
            save_name = '{}/{}_{}.pth'.format(self.model_dir, "checkpoint", self.iteration)
            print("Received signal {}. \nSaving model to {}".format(signalNumber, save_name))
            save_checkpoint(self.epoch, self.ious.avg, self.losses.avg, self.model, self.optimiser, save_name, is_train=False,
                            scheduler=self.lr_schedulers)
        synchronize()
        cleanup_env()
        exit(1)


def register_interrupt_signals(trainer):
    #for i in [x for x in dir(signal) if x.startswith("SIG")]:
    #    try:
    #        signum = getattr(signal, i)
    #        if signum != 0:
    #            print("Signal number: {}, {}".format(signum, i))
    #            signal.signal(signum, trainer.backup_session)
    #    except (OSError, RuntimeError) as m:  # OSError for Python3, RuntimeError for 2
    #        print("Skipping {}".format(i))
    signal.signal(signal.SIGHUP, trainer.backup_session)
    signal.signal(signal.SIGINT, trainer.backup_session)
    signal.signal(signal.SIGQUIT, trainer.backup_session)
    signal.signal(signal.SIGILL, trainer.backup_session)
    signal.signal(signal.SIGTRAP, trainer.backup_session)
    signal.signal(signal.SIGABRT, trainer.backup_session)
    signal.signal(signal.SIGBUS, trainer.backup_session)
    #signal.signal(signal.SIGKILL, trainer.backup_session)
    signal.signal(signal.SIGALRM, trainer.backup_session)
    signal.signal(signal.SIGTERM, trainer.backup_session)


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists("runs/" + args.network_name) and is_main_process():
        os.makedirs("runs/" + args.network_name)
    trainer = Trainer(args)
    register_interrupt_signals(trainer)
    trainer.start()
    trainer.backup_session(signal.SIGQUIT, None)
    synchronize()
    cleanup_env()


