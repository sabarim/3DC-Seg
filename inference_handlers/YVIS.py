import os
import cv2
import torch
import numpy as np
import yaml
from PIL import Image
from scipy.misc import imresize
from torch.utils.data import DataLoader
from torch.nn import functional as F
from inference_handlers.SpatialEmbInference import forward
from inference_handlers.infer_utils.LinearTubeStitching import stitch_clips_best_overlap
from inference_handlers.infer_utils.Visualisation import create_color_map, annotate_instance
from inference_handlers.infer_utils.output_utils import get_class_label
from loss.SpatiotemporalEmbUtils import Cluster
from utils.AverageMeter import AverageMeter
from utils.Constants import PRED_LOGITS, PRED_SEM_SEG, PRED_EMBEDDING

cluster = Cluster()
# number of overlapping frames for stitching
OVERLAPS = 3

with open('/globalwork/mahadevan/vision/davis-unsupervised/inference_handlers/infer_utils/data/youtube_vis_category_names.yaml', 'r') as fh:
  CATEGORY_NAMES = yaml.load(fh, Loader=yaml.SafeLoader)

def infer_yvis(dataset, model, criterion, writer, args, distributed=False):
  losses = AverageMeter()
  ious = AverageMeter()
  # switch to evaluate mode
  model.eval()

  with torch.no_grad():
    for seq in dataset.get_video_ids():
    # for seq in ['libby',
    #             "loading","mbike-trick","motocross-jump","paragliding-launch","parkour","pigs","scooter-black","shooting","soapbox" ]:
    # for seq in ['lab-coat']:
      ious_per_video = AverageMeter()
      dataset.set_video_id(seq)
      test_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False) if distributed else None
      dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False, sampler=test_sampler, pin_memory=True)

      stitched_instance_map = None
      all_instance_pred = None
      all_semantic_pred = None
      all_images = None
      for iter, input_dict in enumerate(dataloader):
        if not args.exhaustive and (iter % (args.tw - OVERLAPS)) != 0:
          continue
        info = input_dict['info']
        if iter == 0:
          shape = tuple(info['num_frames'][0].int().numpy(), ) + tuple(input_dict['images'].shape[-2:], )
          all_instance_pred = torch.zeros(shape).int()
          all_semantic_pred = torch.zeros(shape).int()
          all_images = torch.zeros(shape).unsqueeze(-1).repeat(1,1,1,3).float()

        clip_frames = info['support_indices'][0].data.cpu().numpy()
        batch_size = input_dict['images'].shape[0]

        assert batch_size == 1
        all_images[clip_frames] = input_dict['images'][0].permute(1,2,3,0).float()
        input, input_var, loss, pred_dict = forward(criterion, input_dict, ious, model, args)
        losses.update(loss)
        pred_mask = F.softmax(pred_dict[PRED_LOGITS], dim=1)
        pred_multi = F.softmax(pred_dict[PRED_SEM_SEG][:, :-1], dim=1) if PRED_SEM_SEG in pred_dict else None
        pred_embedding = pred_dict[PRED_EMBEDDING]

        seed_map = torch.argmax(pred_mask, dim=1).float() * pred_mask[:, -1]
        if (args.embedding_dim - 4) % 3 == 0:
          seed_map = torch.argmax(pred_mask, dim=1).float() * torch.sigmoid(pred_embedding[:, -1])
          pred_embedding[:, -1] = seed_map
          pred_spatemb = pred_embedding.data.cpu()
        else:
          pred_spatemb = torch.cat((pred_embedding.cuda(), seed_map.unsqueeze(1).float().cuda()), dim=1)

        instance_map, predictions, vis = cluster.cluster(pred_spatemb[0], threshold=0.5, n_sigma=args.n_sigma,
                                                         iou_meter = None, in_mask=None,
                                                         visualise_clusters=args.visualise_clusters,
                                                         floating= np.any(['floating' in loss for loss in args.losses]))

        if stitched_instance_map is not None:
          stitched_instance_map = stitch_clips_best_overlap(stitched_instance_map, instance_map.data.cpu(), OVERLAPS)
          all_instance_pred[clip_frames[OVERLAPS:]] = stitched_instance_map.data.int().cpu()[OVERLAPS:].int()
          all_semantic_pred[clip_frames[OVERLAPS:]] = torch.argmax(pred_multi, dim=1).data.cpu().int()[0][OVERLAPS:]
        else:
          stitched_instance_map = instance_map.data.cpu()
          all_instance_pred[clip_frames] = stitched_instance_map.data.int().cpu()
          all_semantic_pred[clip_frames] = torch.argmax(pred_multi, dim=1).data.cpu().int()[0]

        save_results(all_instance_pred[clip_frames],all_semantic_pred[clip_frames],all_images[clip_frames],
                     info, os.path.join('results', args.network_name, 'per_clip'), per_clip=True)

      ious.update(ious_per_video.avg)
      # save_results(all_instance_pred, all_semantic_pred, all_images, info, os.path.join('results', args.network_name))
      print('Sequence {}\t IOU {iou}'.format(input_dict['info']['name'], iou=ious_per_video.avg))

  print('Finished Inference Loss {losses.avg:.5f} IOU {iou.avg: 5f}'
        .format(losses=losses, iou=ious))


def save_results(pred, classes, images, info, results_path, per_clip=False):
  results_path_annotated = os.path.join(results_path, 'vis', info['name'][0])
  results_path = os.path.join(results_path, info['name'][0])
  pred = pred.data.cpu().numpy().astype(np.uint8)
  cmap = create_color_map().tolist()

  (lh, uh), (lw, uw) = info['pad']
  h, w = pred.shape[-2:]
  l = list(range(len(pred))) if not per_clip else info['support_indices'][0].data.cpu().numpy()
  for f in range(len(pred)):
    image = images[f, lh[0]:h - uh[0], lw[0]:w - uw[0]]
    category_mask = classes[f, lh[0]:h - uh[0], lw[0]:w - uw[0]]
    M = pred[f, lh[0]:h - uh[0], lw[0]:w - uw[0]]
    instance_ids = np.setdiff1d(np.unique(M), [0])
    category_ids = []
    image_annotated = (image*255.0).int().data.numpy()

    for instance_id in instance_ids:
      colour = cmap[instance_id]
      label = get_class_label((M == instance_id), category_mask)
      if label in CATEGORY_NAMES:
        category_name = CATEGORY_NAMES[label]
      else:
        category_name = str(label)
      instance = (M == instance_id).astype(np.uint8)
      image_annotated = annotate_instance(image_annotated, instance, colour, category_name)
      category_ids += [label]

    if not os.path.exists(results_path):
      os.makedirs(results_path)
    if not os.path.exists(results_path_annotated):
      os.makedirs(results_path_annotated)
    cv2.imwrite(os.path.join(results_path_annotated, '{:05d}.png'.format(l[f])), image_annotated)




    img_M = Image.fromarray(M)
    # img_M.putpalette(palette)

    # image_annotated = Image.fromarray(image_annotated)


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