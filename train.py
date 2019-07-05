import importlib
import inspect
import os
import random
import time
from collections import OrderedDict

import numpy as np
from PIL import Image
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.DAVIS import DAVIS
from models.model import *
# Constants
from utils.utils import font, ToCudaVariable, DAVIS_ROOT, get_best_match, iou, ToLabel, get_centre, \
  iou_simple
from utils.Argparser import parse_args

NUM_EPOCHS = 400
TRAIN_KITTI = False
MASK_CHANGE_THRESHOLD = 1000

BBOX_CROP = True
RANDOM_INSTANCE = True
BEST_IOU=0

network_models = {0:"RGMP", 1:"Similarity", 2:"FeatureWarpModel", 3:"FeatureWarpedGCModel", 4:"FeatureWarpedGCModelV2",
                  5:"DeformConvModel", 6:"DeformConvModelV2", 7:"FeatureWarpedGCModelV3", 8:"RGMPNonLocal",
                  9: "RGMPPrediction_Weighting", 10: "AffinityNetwork", 11: "SimilarityNetwork"}
palette = Image.open(DAVIS_ROOT + '/Annotations/480p/bear/00000.png').getpalette()

def Propagate_MS(ms, model, F2, P2, PC, ref_mask):
    h, w = F2.size()[2], F2.size()[3]

    msv_F2, msv_P2 = ToCudaVariable([F2, P2])
    r5, r4, r3, r2  = model.Encoder(msv_F2, msv_P2)
    if args.network == 10:
      # ref_mask = F.interpolate(ref_mask.unsqueeze(1), size=r5.shape[-2:])
      c_ref = get_centre(ref_mask.unsqueeze(1))
      e2 = model.Decoder(r5, ms, r4, r3, r2, PC, c_ref)
    else:
      e2 = model.Decoder(r5, ms, r4, r3, r2)

    return (F.softmax(e2[0], dim=1), r5, e2[-1])


def save_results(all_E, info, num_frames, path):
  if not os.path.exists(path):
    os .makedirs(path)
  for f in range(num_frames):
    E = all_E[0, :, f].numpy()
    # make hard label
    E = ToLabel(E)

    (lh, uh), (lw, uw) = info['pad']
    E = E[lh[0]:-uh[0], lw[0]:-uw[0]]

    img_E = Image.fromarray(E)
    img_E.putpalette(palette)
    img_E.save(os.path.join(path, '{:05d}.png'.format(f)))


def load_weights(model, optimizer):
    start_epoch = 0
    # load saved model if specified
    if args.loadepoch is not None:
      print('Loading checkpoint {}@Epoch {}{}...'.format(font.BOLD, args.loadepoch, font.END))
      if args.loadepoch == '0':
        # transform, checkpoint provided by RGMP
        load_name = 'weights.pth'
        state = model.state_dict()
        checkpoint = torch.load(load_name)
        checkpoint = {"model": OrderedDict([(k.replace("module.", ""), v) for k, v in checkpoint.items()])}
        checkpoint['epoch'] = 0
      else:
        load_name = os.path.join(MODEL_DIR,
                                 '{}.pth'.format(args.loadepoch))
        state = model.state_dict()
        checkpoint = torch.load(load_name)
        start_epoch = checkpoint['epoch'] + 1

      checkpoint_valid = {k: v for k, v in checkpoint['model'].items() if k in state and state[k].shape == v.shape}
      missing_keys = np.setdiff1d(list(state.keys()),list(checkpoint_valid.keys()))
      for key in missing_keys:
        checkpoint_valid[key] = state[key]

      model.load_state_dict(checkpoint_valid)
      if 'optimizer' in checkpoint.keys():
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
      if 'optimizer_extra' in checkpoint.keys():
        optimizer.load_state_dict(checkpoint['optimizer_extra'])
      if 'pooling_mode' in checkpoint.keys():
        POOLING_MODE = checkpoint['pooling_mode']
      del checkpoint
      torch.cuda.empty_cache()
      print('Loaded weights from {}'.format(load_name))

    return model, optimizer, start_epoch


def run_eval(model, epoch):
  # testing
  with torch.no_grad():
    print('[Val] Epoch {}{}{}'.format(font.BOLD, epoch, font.END))
    model.eval()
    losses = []
    ious = []
    start_frame = 0
    seqs = Testset.get_video_ids()
    # seqs = ['bmx-trees']
    for seq in seqs:
      Testset.set_video_id(seq)
      dataloader = DataLoader(Testset, batch_size=1, shuffle=False, num_workers=2)
      ious_per_video = []
      ious_extra_per_video = []
      loss = 0
      all_E = None
      ms = {}
      object_category = {}
      prev_proposals = None
      ms_bg = None

      pbar = tqdm(total=len(dataloader))
      for i, (all_F, all_M, proposals, proposal_centres, info) in enumerate(dataloader):
        all_F, all_M, proposals = all_F[0], all_M[0], proposals[0]
        pbar.update(1)
        seq_name = info['name'][0]
        num_frames = info['num_frames'][0]
        num_objects = info['num_objects'][0]
        proposal_scores = info["proposal_scores"][0]
        proposal_categories = info["proposal_categories"][0]
        freeze_update = np.zeros(num_objects).astype(np.bool)
        iou_per_frame = []
        iou_extra_per_frame = []

        if all_E is None:
          B, C, H, W = all_M.shape
          all_E = torch.zeros(B, num_objects + 1, num_frames, H, W)

        f = i - start_frame - 1

        if i != start_frame and ms_bg is None:
          all_E[:, 0, 0] = torch.where(torch.sum(all_E[:, :, 0], dim=1) == 0,
                                       torch.ones_like(all_E[:, 0, 0]),
                                       torch.zeros_like(all_E[:, 0, 0]))
          mask = torch.sum(all_E[:, 1:, 0], dim=1).repeat(2,1,1) if args.network == 9 else torch.sum(all_E[:, 1:, 0], dim=1)
          msv_F1, bg_P1 = ToCudaVariable([all_F, mask])
          ms_bg = model.Encoder(msv_F1, bg_P1)[0]

        if ms_bg is not None:
          mask = torch.sum(all_E[:, 1:, f], dim=1).repeat(2,1,1) if args.network == 9 else torch.sum(all_E[:, 1:, f], dim=1)
          ref_mask = torch.sum(all_E[:, 1:, 0], dim=1).repeat(2,1,1) \
            if args.network == 9 else torch.sum(all_E[:, 1:, 0], dim=1)
          output, _, pred_deform = Propagate_MS(ms_bg, model, all_F.cuda(), mask.cuda(), proposal_centres,ref_mask)
          all_E[:, 0, f + 1] = 1 - output[:, -1]

        for object in range(num_objects):
          if i == start_frame:
            proposal_selected = get_best_match(all_M[0, object:object+1], proposals[0], num_proposals=info['num_proposals'][0],
                                               object_categories=info['proposal_categories'][0])
            all_E[:, object+1, 0] = proposal_selected[0] if proposal_selected[0] is not None else all_E[:, 0, 0]

            object_category[object] = proposal_selected[-1] if proposal_selected[-1] is not None else -1
            image = all_F
            mask = all_E[:, object+1, 0]
            if args.network == 9:
              mask = mask.repeat(2, 1, 1)
            msv_F1, msv_P1, all_M = ToCudaVariable([image, mask, all_M])
            ms[object] = model.Encoder(msv_F1, msv_P1)[0]
          else:
            predictions = all_E[:, object + 1, f]
            proposal_selected = get_best_match((torch.argmax(all_E[:, :, f], dim=1) == object + 1).int(), prev_proposals,
                                               num_proposals=info['num_proposals'][0],
                                               object_categories=info['proposal_categories'][0])

            if proposal_selected[0] is not None:
              if args.network == 9:
                predictions = torch.cat((predictions, proposal_selected[0][None].float()), dim=0)
              else:
                # predictions = proposal_selected[0][None].float() * 0.75 + all_E[:, 0, f] * 0.25
                predictions[proposal_selected[0][None] == 1] = 1
            elif args.network == 9:
              predictions = torch.cat((predictions, torch.zeros_like(predictions)), dim=0)

            output, _, pred_extra = Propagate_MS(ms[object], model, all_F, predictions,
                                                  proposal_centres, all_E[:, object + 1, 0])
            loss_extra, iou_extra = process_extra(pred_extra, all_M[:, object:object + 1], all_E[:, object+1, 0],proposals, info)
            all_E[:, object + 1, f + 1] = output[:, -1].detach()
            loss = loss + criterion(output[:, -1].permute(1, 2, 0), all_M[:, object:object + 1].cuda().float())

            iou_per_frame += [iou(torch.cat((1 - all_E[:, object + 1:object + 2, f + 1],
                                           all_E[:, object + 1:object + 2, f + 1]), dim=1),
                                all_M[:, object:object + 1])]
            iou_extra_per_frame += [iou_extra]

            change_in_pixels = abs((torch.argmax(all_E, dim=1)[0, f] == object + 1).sum() -
                                   (torch.argmax(all_E, dim=1)[0, f - 1] == object + 1).sum())

            if args.update_refs and proposal_selected[0] is not None and proposal_selected[1] > 0.15 and not freeze_update[object] and \
              proposal_selected[-1] == object_category[object]:
              if change_in_pixels.numpy() < MASK_CHANGE_THRESHOLD:
                print("Updating object {} in frame {} as reference".format(object+1, f))
                ms[object] = model.Encoder(prev_frame.cuda(), proposal_selected[0][None].repeat(2,1,1).cuda())[0]
              else:
                freeze_update[object] = True
            # all_E[:, 0, f+1] = bg
        # store the previous proposals to compute the best match
        prev_proposals = proposals[0]
        prev_frame = all_F

        if i!=start_frame:
          ious_per_video += [np.mean(iou_per_frame)]
          ious_extra_per_video += [np.mean(iou_extra_per_frame)]

      pbar.close()

      print("[{}]:- loss:{}, iou: {}, iou_extra:{}".format(seq, loss.data.cpu().numpy() / len(dataloader),
                                                           np.mean(ious_per_video), np.mean(ious_extra_per_video)))
      losses += [loss.data.cpu().numpy() / len(dataloader)]
      ious += [np.mean(ious_per_video)]

      if args.results_path is not None:
        save_results(all_E, info, num_frames, os.path.join(args.results_path, seq))

    iou_mean = np.mean(ious)
    print("Validation epoch {}:- loss:{}, iou: {}".format(epoch, np.mean(losses), iou_mean))
    writer.add_scalar('Val/BCE', np.mean(losses), epoch)
    writer.add_scalar('Val/IOU', np.mean(iou_mean), epoch)

    save_checkpoint(epoch, iou_mean, model)


def save_checkpoint(epoch, iou_mean, model):
  global BEST_IOU
  if iou_mean > BEST_IOU:
    BEST_IOU = iou_mean
    save_name = '{}/{}.pth'.format(MODEL_DIR, "model_best")
    if not os.path.exists(MODEL_DIR):
      os.makedirs(MODEL_DIR)
    torch.save({'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                },
               save_name)
    print("Saving epoch {} with IOU {}".format(epoch, iou_mean))


def process_extra(pred_extra, all_M, ref_mask, proposals, info, save=False):
  loss_extra = 0
  if args.network == 10:
    proposal_selected = get_best_match(all_M[0], proposals[0], num_proposals=info['num_proposals'][0],
                                       object_categories=info['proposal_categories'][0])
    target_id = proposal_selected[2]
    label_shape = np.array(pred_extra.shape)
    # add an extra column for labelling the association as background
    label_shape[-1]+=1
    label = torch.zeros(tuple(label_shape)).cuda().float()
    pred = torch.zeros(tuple(label_shape)).cuda().float()
    pred[..., 1:] = pred_extra
    pred[..., 0] = 10
    pred = torch.log(F.softmax(pred, dim=-1))
    # target_id is -1 if no proposals are selected
    label[..., target_id + 1] = 1
    # loss_extra = criterion(F.softmax(pred_extended, dim=-1), label.float())
    loss_extra = (-pred * label).sum()
  elif args.network == 11:
    ref_mask = F.interpolate(ref_mask.unsqueeze(1), size=pred_extra.shape[-2:])
    target = F.interpolate(all_M.float(), size=pred_extra.shape[-2:])
    # make One-hot channel
    oh_masks = torch.zeros((pred_extra.shape[0],) + (2,) + tuple(pred_extra.shape[-2:]))
    for o in range(2):
      oh_masks[:, o, :, :] = (ref_mask == o)[:, 0].int()
    pred = torch.bmm(pred_extra.reshape(pred_extra.shape[0],pred_extra.shape[1], -1),
                      oh_masks.reshape((tuple(oh_masks.shape[:2],)) + (-1,)).permute(0,2,1).float().cuda())
    pred = F.softmax(pred.permute(0,2,1).reshape(oh_masks.shape), dim=1)
    loss_extra = criterion(pred[:, -1:].float().cuda(), target.float().cuda())
    pred_upscaled = F.interpolate(pred, size = all_M.shape[-2:])
    proposal_selected = get_best_match(torch.argmax(pred_upscaled, dim=1), proposals[0],
                                       num_proposals=info['num_proposals'][0],
                                       object_categories=info['proposal_categories'][0])
    if proposal_selected[0] is not None:
      iou_extra = iou_simple(proposal_selected[0], all_M[0,0])
    else:
      iou_extra = iou_simple(torch.argmax(pred_upscaled.data.cpu(), dim=1), all_M[0,0])

  return loss_extra, iou_extra


def run_train(model, epoch):
  # TODO: why is that necessary? with model.eval() it doesn't work in a good way
  # model.train()
  model.eval()
  losses = []
  losses_extra = []
  ious = []
  ious_extra = []
  seqs = Trainset.get_video_ids()
  Trainset.remove_gt_proposal = args.remove_gt_proposal
  print('[Train] Epoch {}{}{}'.format(font.BOLD, epoch, font.END))
  for seq in seqs:
    optimizer.zero_grad()
    Trainset.set_video_id(seq)
    dataloader = DataLoader(Trainset, batch_size=1, shuffle=False, num_workers=2)

    tt = time.time()
    num_bptt = args.bptt_len
    loss = 0
    loss_extra = 0
    counter = 0
    loss_to_print = 0
    prev_proposals = None
    prev_gt_mask = None
    ms = None
    iou_per_video = []
    iou_extra_per_video = []

    for i, (all_F, all_M, proposals, proposal_centres, info) in enumerate(dataloader):
      all_F, all_M, proposals = all_F[0], all_M[0], proposals[0]
      seq_name = info['name'][0]
      num_objects = info['num_objects'][0]

      if i == 0:
        for _ in range(num_objects):
          object = random.randint(0, num_objects - 1)
          if all_M[:, object:object + 1].sum() > 0:
            break
        B, C, H, W = all_M.shape
        all_E = torch.zeros(B, 1, args.bptt_len, H, W)
        all_M = all_M[:, object:object + 1]
        proposal_selected = get_best_match(all_M[0], proposals[0], num_proposals=info['num_proposals'][0],
                                           object_categories=info['proposal_categories'][0])
        all_E[:, 0, 0] = proposal_selected[0] if proposal_selected[0] is not None else all_M[0]

        image = all_F
        mask = all_E[:, 0, 0]
        if args.network == 9:
          mask = mask.repeat(2,1,1)
        msv_F1, msv_P1, all_M = ToCudaVariable([image, mask, all_M])
        ms = model.Encoder(msv_F1, msv_P1)[0]
        # counter+=1
      else:
        all_M = all_M[:, object:object+1]
        # f = i - start_frame - 1
        f = i - 1
        predictions = all_E[:, 0, f]
        # use the ground truth match during training
        proposal_selected = get_best_match(prev_gt_mask, prev_proposals,
                                           num_proposals=info['num_proposals'][0],
                                           object_categories=info['proposal_categories'][0])
        # proposal_selected = get_best_match((predictions > (1-predictions)).int(), prev_proposals,
        #                                    num_proposals=info['num_proposals'][0],
        #                                    object_categories=info['proposal_categories'][0])

        if proposal_selected[0] is not None:
          if args.network == 9:
            predictions = torch.cat((predictions, proposal_selected[0][None].float()), dim=0)
          elif args.network != 10:
            # predictions = proposal_selected[0][None].float() * 0.75 + all_E[:, 0, f] * 0.25
            predictions[proposal_selected[0][None] == 1] = 1
        elif args.network == 9:
          predictions = torch.cat((predictions, torch.zeros_like(predictions)), dim=0)
            # predictions[proposal_selected[0][None] == 1] = 1

        output, e, pred_extra = Propagate_MS(ms, model, all_F, predictions, proposal_centres, all_E[:, 0, 0])
        # train additional losses such as an affinity matrix
        loss_extra, iou_extra = process_extra(pred_extra, all_M, all_E[:, 0, 0], proposals, info)
        if loss_extra > 0:
          optimizer.zero_grad()
          loss_extra.backward(retain_graph=True)
          optimizer.step()

        all_E[:, 0, f + 1] = output[:, -1].detach()
        loss = loss + criterion(output[:, -1].permute(1, 2, 0), all_M.cuda().float()) * args.loss_scale
        losses_extra += [loss_extra.data.cpu()]
        iou_per_frame = iou(output, all_M)
        iou_per_video += [iou_per_frame]
        iou_extra_per_video += [iou_extra]
        counter += 1
        # logging and display

        # TODO(Paul): get rid of this?
        if (f + 1) % args.bptt_step == 0:
          optimizer.zero_grad()
          # if loss_extra > 0:
          #   loss += loss_extra
          loss.backward(retain_graph=True)
          optimizer.step()
          output.detach()
          if f < num_bptt - 2:
            loss_to_print += loss.data.cpu().float() / max(counter, 1)
            loss = 0
            counter = 0
      prev_proposals = proposals[0]
      prev_gt_mask = all_M


    if loss > 0:
      optimizer.zero_grad()
      # if loss_extra > 0:
      #   loss += loss_extra
      loss.backward()
      optimizer.step()
      loss_to_print += loss.data.cpu().float()

    print('[Epoch {} Sequence {}] loss: {} loss_extra: {} iou: {} iou_extra: {}'.format(epoch, seq_name, loss_to_print / max(counter, 1),
                                                               loss_extra / max(counter, 1), np.mean(iou_per_video), np.mean(iou_extra_per_video)))
    losses += [(loss / max(counter, 1)).detach().cpu().numpy()]
    ious += [np.mean(iou_per_video)]
    ious_extra += [np.mean(iou_extra_per_video)]

  print("epoch {}:- loss:{}, loss_extra: {}, iou: {} iou_extra: {}".format(epoch, np.mean(losses), np.mean(losses_extra), np.mean(ious), np.mean(ious_extra)))
  writer.add_scalar('Train/BCE', np.mean(losses), epoch)
  writer.add_scalar('Train/Extra', np.mean(losses_extra), epoch)
  writer.add_scalar('Train/IOU', np.mean(ious), epoch)
  return np.mean(losses), np.mean(ious)


if __name__ == '__main__':
    args = parse_args()
    MODEL_DIR = os.path.join('saved_models', args.network_name)
    print("Arguments used: {}".format(args))

    Trainset = DAVIS(DAVIS_ROOT, imset='2017/train.txt', is_train=True, random_instance=RANDOM_INSTANCE,
                     multi_object=args.MO, bptt_len=args.bptt_len,
                     proposal_dir='/globalwork/mahadevan/mywork/data/DAVIS17/DAVIS/maskrcnn-proposals/davis-train/')
    Testset = DAVIS(DAVIS_ROOT, imset='2017/val.txt')

    module = importlib.import_module("models.model")
    class_ = getattr(module, network_models[args.network])
    n_classes = 10 if args.MO else 2
    sig = inspect.signature(class_.__init__)
    if "n_classes" in sig.parameters.keys():
        sig.bind(n_classes)
        model = class_(n_classes)
    else:
        model = class_()
    print("Using model: {}".format(model.__class__))
    print("Using parameters: bbox crop {}, pred averaging {} predict deform conv {}, model directory {} "
          "random instance {}".format(BBOX_CROP, False, False, MODEL_DIR, RANDOM_INSTANCE))
    if torch.cuda.is_available():
        model.cuda()

    # print(summary(model, tuple((256,256)), batch_size=1))
    writer = SummaryWriter(log_dir="runs/" + MODEL_DIR)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer_extra = torch.optim.Adam(model.parameters(), lr=args.lr)
    model, optimizer, start_epoch = load_weights(model, optimizer)# params

    params = []
    for key, value in dict(model.named_parameters()).items():
      if value.requires_grad:
        params += [{'params':[value],'lr':args.lr, 'weight_decay': 4e-5}]

    criterion = torch.nn.BCELoss()
    # iters_per_epoch = len(Trainloader)
    model.eval()
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95) if args.adaptive_lr else None

    if args.task == "train":
      best_loss = 0
      best_iou = 0
      for epoch in range(start_epoch, args.num_epochs):
        loss_mean, iou_mean  = run_train(model, epoch)
        if lr_scheduler is not None:
          lr_scheduler.step(epoch)
        if iou_mean > best_iou or loss_mean < best_loss:
          save_checkpoint(epoch, iou_mean, model)

        if (epoch + 1) % args.eval_epoch == 0:
          run_eval(model, epoch)
    else:
      run_eval(model, 0)

