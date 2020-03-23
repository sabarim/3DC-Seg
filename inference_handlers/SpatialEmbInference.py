import os

import numpy as np
import torch
from PIL import Image
from scipy.misc import imresize
from torch.nn import functional as F
# cluster module
from torch.utils.data import DataLoader

from Forward import format_pred
from inference_handlers.infer_utils.LinearTubeStitching import stitch_clips_best_overlap
from inference_handlers.infer_utils.StitchWithGT import stitch_with_gt
from inference_handlers.infer_utils.Visualisation import visualize_embeddings
from loss.SpatiotemporalEmbUtils import Cluster, Visualizer
from network.NetworkUtil import run_forward
from utils.AverageMeter import AverageMeter
from utils.Constants import DAVIS_ROOT, PRED_LOGITS, PRED_SEM_SEG, PRED_EMBEDDING

cluster = Cluster()
# number of overlapping frames for stitching
OVERLAPS = 3
SEED_THRESHOLD = 0.8


def infer_spatial_emb(dataset, model, criterion, writer, args, distributed=False):
  losses = AverageMeter()
  ious = AverageMeter()
  # switch to evaluate mode
  model.eval()
  palette = Image.open(DAVIS_ROOT + '/Annotations_unsupervised/480p/bear/00000.png').getpalette()

  with torch.no_grad():
    for seq in dataset.get_video_ids():
    # for seq in ['libby',
    #             "loading","mbike-trick","motocross-jump","paragliding-launch","parkour","pigs","scooter-black","shooting","soapbox" ]:
    # for seq in ['blackswan']:
      ious_per_video = AverageMeter()
      dataset.set_video_id(seq)
      test_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False) if distributed else None
      dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False, sampler=test_sampler, pin_memory=True)

      stitched_instance_map = None
      all_instance_pred = None
      all_semantic_pred = None
      for iter, input_dict in enumerate(dataloader):
        if not args.exhaustive and (iter % (args.tw - OVERLAPS)) != 0:
          continue
        info = input_dict['info']
        if iter == 0:
          shape = tuple(info['num_frames'].int().numpy(), ) + tuple(input_dict['images'].shape[-2:], )
          all_instance_pred = torch.zeros(shape).int()
          all_semantic_pred = torch.zeros(shape).int()

        target = input_dict['target_extra']['similarity_raw_mask']
        instances_one_hot = input_dict['target_extra']['similarity_ref'][:, 1:].squeeze()
        batch_size = input_dict['images'].shape[0]
        input, input_var, loss, pred_dict = forward(criterion, input_dict, ious, model, args)
        losses.update(loss)
        pred_mask = F.softmax(pred_dict[PRED_LOGITS], dim=1)
        pred_multi = F.softmax(pred_dict[PRED_SEM_SEG], dim=1) if PRED_SEM_SEG in pred_dict else None
        pred_embedding = pred_dict[PRED_EMBEDDING]

        seed_map = torch.argmax(pred_mask, dim=1).float() * pred_mask[:, -1]
        if (args.embedding_dim - 4) % 3 == 0:
          # seed_map = torch.argmax(pred_mask, dim=1).float() * torch.sigmoid(pred_embedding[:, -1])
          if 'ce' in args.losses:
            print("------------INFO: Using seed map from the embedding head--------------")
            if args.use_fg_mask:
              seed_map = torch.argmax(pred_mask, dim=1).float() *torch.sigmoid(pred_embedding[:, -1])
            else:
              seed_map = torch.sigmoid(pred_embedding[:, -1])
          else:
            print("-------------INFO: Using seed map from the segmentation head.------------")
            seed_map = torch.sigmoid(pred_dict[PRED_LOGITS][:,-1])
          pred_embedding[:, -1] = seed_map
          pred_spatemb = pred_embedding.data.cpu()
        else:
          pred_spatemb = torch.cat((pred_embedding.cuda(), seed_map.unsqueeze(1).float().cuda()), dim=1)

        instance_map, predictions, vis = cluster.cluster(pred_spatemb[0], threshold=SEED_THRESHOLD, n_sigma=args.n_sigma,
                                                         iou_meter = ious_per_video, in_mask=instances_one_hot.data.cpu(),
                                                         visualise_clusters=args.visualise_clusters,
                                                         floating= np.any(['floating' in loss for loss in args.losses]))

        clip_frames = info['support_indices'][0].data.cpu().numpy()
        if args.save_per_clip and pred_embedding is not None:
          save_per_clip(iter, instance_map=instance_map, e= None, info=info,
                        results_path=os.path.join('results', args.network_name,'per_clip'), palette=palette)
          if pred_multi is not None:
            save_per_clip(iter, instance_map=torch.argmax(pred_multi[0].data.cpu(), dim=0), e=None, info=info,
                          results_path=os.path.join('results', args.network_name, 'sem_seg'), palette=palette)
          if vis is not None:
            clusters=vis[0]
            img = Image.fromarray(clusters)
            dir= os.path.join('results', args.network_name, 'cluster_vis', info['name'][0])
            if not os.path.exists(dir):
              os.makedirs(dir)
            img.save(os.path.join(dir,'clip_{:05d}_{:05d}.png'.format(iter, iter + 7)))
        # if args.visualise_clusters:
        #   embedding_dim = args.embedding_dim - args.n_sigma - 1
        #   vis_clusters, = visualize_embeddings(pred_embedding[:, :embedding_dim].reshape(embedding_dim,-1).permute(1,0).data.cpu(),
        #                                        target.reshape(-1).data.cpu(),
        #                                        pred_embedding[:, embedding_dim: embedding_dim + args.n_sigma].
        #                                        reshape(args.n_sigma, -1).permute(1,0).data.cpu(),
        #                                        True)


        assert batch_size == 1
        if args.stitch == 'gt':
          stitched_instance_map = stitch_with_gt(instance_map, target, info['num_objects'][0].item())
          all_instance_pred[clip_frames] = stitched_instance_map.data.int().cpu()
          all_semantic_pred[clip_frames] = torch.argmax(pred_mask, dim=1).data.cpu().int()[0]
        elif stitched_instance_map is not None:
          stitched_instance_map = stitch_clips_best_overlap(stitched_instance_map, instance_map.data.cpu(), OVERLAPS)
          all_instance_pred[clip_frames[OVERLAPS:]] = stitched_instance_map.data.int().cpu()[OVERLAPS:].int()
          all_semantic_pred[clip_frames[OVERLAPS:]] = torch.argmax(pred_mask, dim=1).data.cpu().int()[0][OVERLAPS:]
        else:
          stitched_instance_map = instance_map.data.cpu()
          last_predictions = predictions
          all_instance_pred[clip_frames] = stitched_instance_map.data.int().cpu()
          all_semantic_pred[clip_frames] = torch.argmax(pred_mask, dim=1).data.cpu().int()[0]
        # for i in range(input_dict['images'].shape[2]):
        #   visualise(input_dict, instance_map, target, pred_spatemb, i, args, iter)
          # save_results(input_dict['images'], predictions, info, args)
        #all_instance_pred[info['support_indices'][0]] = stitched_instance_map.int()
        #all_semantic_pred[info['support_indices'][0]] = torch.argmax(pred_mask, dim=1).data.cpu().int()[0]


      ious.update(ious_per_video.avg)
      save_results(all_instance_pred, info, os.path.join('results', args.network_name), palette)
      save_results(all_semantic_pred, info, os.path.join('results', args.network_name, 'semantic_pred'), palette)
      print('Sequence {}\t IOU {iou}'.format(input_dict['info']['name'], iou=ious_per_video.avg))

  print('Finished Inference Loss {losses.avg:.5f} IOU {iou.avg: 5f}'
        .format(losses=losses, iou=ious))


def visualise(input_dict, instance_map, instances, pred_extra, i, args, iter):
  # Visualizer
  visualizer = Visualizer(('image', 'pred', 'sigma', 'seed'))
  results_path = os.path.join('results', args.network_name, input_dict['info']['name'][0])
  if not os.path.exists(results_path):
    os.makedirs(results_path)
  base, _ = os.path.splitext(os.path.basename(input_dict['info']['name'][0]))

  im = input_dict['images'][0, :, i]
  instances = instances.squeeze()[i]
  instance_map = instance_map[i]
  # visualizer.display(im, 'image')
  visualizer.savePlt([instance_map.cpu(), instances.cpu()], 'pred', os.path.join(results_path, base +
                                                                                 '_{:05d}_pred.png'.format(iter+i)))
  sigma = pred_extra[0, 3:-1, i].data.cpu()
  sigma = (sigma - sigma.min()) / (sigma.max() - sigma.min())
  sigma[:, instances == 0] = 0
  count=0
  for s in sigma:
    count+=1
    visualizer.savePlt(s.cpu(), 'sigma', os.path.join(results_path, base +
                                                          '_{:05d}_sigma_{:1d}.png'.format(iter+i, count)))
  seed = torch.sigmoid(pred_extra[0, -1, i].data.cpu())
  visualizer.savePlt(seed, 'seed', os.path.join(results_path, base +
                                                '_{:05d}_seed.png'.format(iter+i)))


def forward(criterion, input_dict, ious, model, args):
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
  pred = format_pred(pred)
  pred_dict = dict([(key, F.interpolate(val, target.shape[2:], mode="trilinear")) for key,val in pred.items()])
  loss=0

  return input, input_var, loss, pred_dict


def save_results(pred, info, results_path, palette):
  results_path = os.path.join(results_path, info['name'][0])
  pred = pred.data.cpu().numpy().astype(np.uint8)
  (lh, uh), (lw, uw) = info['pad']
  h, w = pred.shape[-2:]
  for f in range(len(pred)):
    M = pred[f, lh[0]:h-uh[0], lw[0]:w-uw[0]]
    shape = info['shape480p'] if 'shape480p' in info else info['shape']
    img_M = Image.fromarray(imresize(M, shape, interp='nearest'))
    img_M.putpalette(palette)
    if not os.path.exists(results_path):
      os.makedirs(results_path)
    img_M.save(os.path.join(results_path, '{:05d}.png'.format(f)))


def save_per_clip(iter, instance_map, e, info, results_path, palette):
  results_path = os.path.join(results_path, info['name'][0])
  e_path = os.path.join(results_path, 'embeddings')
  if not os.path.exists(results_path) and instance_map is not None:
    os.makedirs(results_path)
  if not os.path.exists(e_path) and e is not None:
    os.makedirs(e_path)

  (lh, uh), (lw, uw) = info['pad']

  if e is not None:
    e = e[:, :, :, lh[0]:-uh[0], lw[0]:-uw[0]]
    save_dict = {"embeddings": e, 'frames': info['support_indices'][0].data.cpu()}
    with open(os.path.join(e_path, 'clip_{:05d}_{:05d}.pickle'.format(iter, iter + 7)), 'wb') as f:
      np.pickle.dump(save_dict, f)

  if instance_map is not None:
    for f in range(len(instance_map)):
      h, w = instance_map.shape[-2:]
      M = instance_map[f, lh[0]:h - uh[0], lw[0]:w - uw[0]]
      img_M = Image.fromarray(imresize(M.byte(), info['shape'], interp='nearest'))
      img_M.putpalette(palette)
      if not os.path.exists(results_path):
        os.makedirs(results_path)
      img_M.save(os.path.join(results_path,  'clip_{:05d}_{:05d}_frame_{:05d}.png'.format(iter, iter + 7, f)))


