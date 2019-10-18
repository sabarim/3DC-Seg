import os
import time

import torch
import torchvision
from tqdm import tqdm
from torch.nn import functional as F
from network.NetworkUtil import run_forward
from utils.AverageMeter import AverageMeter


# cluster module
from utils.SpatialEmbUtils import Cluster, Visualizer

cluster = Cluster()


def infer_spatial_emb(dataloader, model, criterion, writer, args):
  losses = AverageMeter()
  ious = AverageMeter()
  # switch to evaluate mode
  model.eval()
  # for seq in dataloader.dataset.get_video_ids():
  for seq in ['dogs-jump']:
    dataloader.dataset.set_video_id(seq)
    for iter, input_dict in enumerate(dataloader):
      if not args.exhaustive and iter % args.tw != 0:
        continue
      info = input_dict['info']
      instances = input_dict['target_extra']['similarity_raw_mask'].squeeze()
      batch_size = input_dict['images'].shape[0]
      input, input_var, loss, pred, pred_extra = forward(criterion, input_dict, ious, model)
      pred_mask = F.softmax(pred, dim=1)
      pred_spatemb = torch.cat((pred_extra.cuda(), pred[:, -1:]), dim=1)
      instance_map, predictions = cluster.cluster(pred_spatemb[0], threshold=0.5)

      assert batch_size == 1
      for i in range(input_dict['images'].shape[2]):
        visualise(input_dict, instance_map, instances, pred_spatemb, i, args, iter)
        save_results(input_dict['images'], predictions, info, args, i)

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
  instances = instances[i]
  instance_map = instance_map[i]
  # visualizer.display(im, 'image')
  visualizer.savePlt([instance_map.cpu(), instances.cpu()], 'pred', os.path.join(results_path, base +
                                                                                 '_{:05d}_pred.png'.format(iter+i)))
  sigma = pred_extra[0, 3, i].data.cpu()
  sigma = (sigma - sigma.min()) / (sigma.max() - sigma.min())
  sigma[instances == 0] = 0
  visualizer.savePlt(sigma.cpu(), 'sigma', os.path.join(results_path, base +
                                                        '_{:05d}_sigma.png'.format(iter+i)))
  seed = torch.sigmoid(pred_extra[0, 4, i].data.cpu())
  visualizer.savePlt(seed, 'seed', os.path.join(results_path, base +
                                                '_{:05d}_seed.png'.format(iter+i)))


def forward(criterion, input_dict, ious, model):
  input = input_dict["images"]
  target = input_dict["target"]
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
  pred_extra = pred[1] if len(pred) > 1 else None
  if len(pred[0].shape) > 4:
    pred = F.interpolate(pred[0], target.shape[2:], mode="trilinear")
    if pred_extra is not None:
      pred_extra = F.interpolate(pred_extra, target.shape[2:], mode="trilinear")
  else:
    pred = F.interpolate(pred[0], target.shape[2:], mode="bilinear")
    pred_extra = F.interpolate(pred_extra, target.shape[2:], mode="bilinear")
  loss = 0

  return input, input_var, loss, pred, pred_extra


def save_results(images, predictions, info, args, i):
  results_path=os.path.join('results', args.network_name, info['name'][0])
  if not os.path.exists(results_path):
    os.makedirs(results_path)
  base, _ = os.path.splitext(os.path.basename(info['name'][0]))
  txt_file = os.path.join(results_path, base + '.txt')

  with open(txt_file, 'w') as f:
    # loop over instances
    for id, pred in enumerate(predictions):
      im_name = base + '_{:05d}_{:02d}.png'.format(i, id)
      im = torchvision.transforms.ToPILImage()(
        pred['mask'][i].unsqueeze(0))

      # write image
      im.save(os.path.join(results_path, im_name))

      # write to file
      cl = 26
      score = pred['score']
      f.writelines("{} {} {:.02f}\n".format(im_name, cl, score))