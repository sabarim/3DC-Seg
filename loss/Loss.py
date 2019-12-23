import torch
import numpy as np
import torch.nn.functional as F
from loss.SpatialEmbLoss import SpatioTemporalEmbLoss, SpatialEmbLoss, CovarianceLoss, CovarianceLossDirect

from loss.embedding_loss import compute_embedding_loss
from utils.AverageMeter import AverageMeter


def bootstrapped_ce_loss(raw_ce, n_valid_pixels_per_im=None, fraction=0.25):
  n_valid_pixels_per_im = raw_ce.shape[-1]*raw_ce.shape[-2] if n_valid_pixels_per_im is None else n_valid_pixels_per_im
  ks = torch.max(torch.tensor(n_valid_pixels_per_im * fraction).cuda().int(), torch.tensor(1).cuda().int())
  if len(raw_ce.shape) > 3:
    bootstrapped_loss = raw_ce.reshape(raw_ce.shape[0], raw_ce.shape[1], -1).topk(ks, dim=-1)[0].mean(dim=-1).mean()
  else:
    bootstrapped_loss = raw_ce.reshape(raw_ce.shape[0], -1).topk(ks, dim=-1)[0].mean(dim=-1).mean()
  return bootstrapped_loss


def compute_loss(args, criterion, pred, target, target_extra=None, iou_meter=None):
  pred_seg = pred[0]
  pred_mask = None
  pred_extra = pred[1] if len(pred) > 1 else None
  loss_mask = 0
  loss_image = torch.zeros_like(pred_seg[:, -1])
  if "ce" in args.losses:
    if len(pred_seg.shape) > 4:
      pred_mask = F.interpolate(pred_seg, target.shape[2:], mode="trilinear")
    else:
      pred_mask = F.interpolate(pred_seg, target.shape[2:], mode="bilinear")

    if isinstance(criterion, torch.nn.CrossEntropyLoss):
      loss_image = criterion(pred_mask, target.squeeze(1).cuda().long())
    else:
      # For binary segmentation use just the first and the last channel
      pred_mask = torch.stack((pred_mask[:, 0], pred_mask[:, -1]), dim=1)
      loss_image = criterion(pred_mask[:, -1], target.squeeze(1).cuda().float())
    loss_mask = bootstrapped_ce_loss(loss_image)

  loss_extra = {}
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
      loss_similarity = criterion_extra(y, similarity_target)
      loss_extra['similarity'] = bootstrapped_ce_loss(loss_similarity)
    if "embedding" in args.losses:
      pred_extra = F.interpolate(pred_extra, scale_factor=(1,8,8), mode='trilinear')
      loss_embedding, _, _ = compute_embedding_loss(pred_extra, target_extra['similarity_ref'].cuda(), args.config_path)
      loss_extra['embedding'] = loss_embedding
    elif "spatiotemporal_embedding" in args.losses:
      iou_all_instances = AverageMeter()
      pred_extra = F.interpolate(pred_extra, size=target.shape[-3:], mode='trilinear')
      # spatial embedding loss expects the last channel to be a seed map, which could be the fg/bg prediction here
      # pred_spatemb = torch.cat((pred_extra.cuda(), pred_mask[:, -1:]), dim=1)
      pred_spatemb = pred_extra
      criterion_extra = SpatioTemporalEmbLoss(n_sigma=args.embedding_dim - 4, to_center=args.coordinate_centre)
      loss_extra['spatiotemporal_embedding'] = criterion_extra.forward(pred_spatemb, target_extra['similarity_raw_mask'].cuda(),
                                                               labels=None, iou=True, iou_meter=iou_all_instances,
                                                               w_var=10)
      iou_meter.update(iou_all_instances.avg)
      # loss_mask = 0
    elif "spatial_embedding" in args.losses:
      iou_all_instances = AverageMeter()
      pred_extra = F.interpolate(pred_extra, scale_factor=(1, 8, 8), mode='trilinear')
      # spatial embedding loss expents the last channel to be a seed map, which could be the fg/bg prediction here
      # pred_spatemb = torch.cat((pred_extra.cuda(), pred_mask[:, -1:]), dim=1)
      pred_spatemb = pred_extra
      criterion_extra = SpatialEmbLoss(n_sigma=args.embedding_dim - 4)
      loss_extra['spatial_embedding'] = criterion_extra(pred_spatemb, target_extra['similarity_raw_mask'].cuda(),
                                         labels=None, iou=True, iou_meter=iou_all_instances)
      iou_meter.update(iou_all_instances.avg)
      # loss_mask = 0

    elif np.any(["covariance" in s for s in args.losses]):
      iou_all_instances = AverageMeter()
      pred_extra = F.interpolate(pred_extra, size=target.shape[-3:], mode='trilinear')
      # spatial embedding loss expents the last channel to be a seed map, which could be the fg/bg prediction here
      # pred_spatemb = torch.cat((pred_extra.cuda(), pred_mask[:, -1:]), dim=1)
      pred_spatemb = pred_extra
      criterion_extra = CovarianceLoss(n_sigma=args.embedding_dim - 4) if 'covariance_loss' in args.losses else \
        CovarianceLossDirect(n_sigma=args.embedding_dim - 4)
      loss_extra['covar_loss'] = criterion_extra(pred_spatemb, target_extra['similarity_raw_mask'].cuda(),
                                         labels=None, iou=True, iou_meter=iou_all_instances)
      iou_meter.update(iou_all_instances.avg)

    if 'multi_class' in args.losses:
      assert pred_seg.shape[1] > 2 and 'sem_seg' in target_extra
      criterion_multi = torch.nn.CrossEntropyLoss(reduce=False)
      # FIXME: assumption that the last channel is always foreground/background segmentation
      loss_multi = criterion_multi(pred_seg[:, :-1], target_extra['sem_seg'].cuda().long().squeeze(1))
      loss_multi = bootstrapped_ce_loss(loss_multi)
      loss_mask += loss_multi
      loss_extra['loss_multi'] = loss_multi



  # print("loss_extra {}".format(loss_extra))
  loss = loss_mask + sum(loss_extra.values())

  return loss, loss_image, pred_mask, loss_extra
