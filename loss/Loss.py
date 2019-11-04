import torch
import torch.nn.functional as F
from loss.SpatialEmbLoss import SpatioTemporalEmbLoss, SpatialEmbLoss

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
  pred_mask = pred[0]
  pred_extra = pred[1] if len(pred) > 1 else None
  loss_mask = 0
  loss_image = torch.zeros_like(pred_mask[:, -1])
  if "ce" in args.losses:
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
      loss_extra, _, _ = compute_embedding_loss(pred_extra, target_extra['similarity_ref'].cuda(), args.config_path)
    elif "spatiotemporal_embedding" in args.losses:
      iou_all_instances = AverageMeter()
      pred_extra = F.interpolate(pred_extra, size=target.shape[-3:], mode='trilinear')
      # spatial embedding loss expects the last channel to be a seed map, which could be the fg/bg prediction here
      # pred_spatemb = torch.cat((pred_extra.cuda(), pred_mask[:, -1:]), dim=1)
      pred_spatemb = pred_extra
      criterion_extra = SpatioTemporalEmbLoss(n_sigma=args.embedding_dim - 4)
      loss_extra = criterion_extra(pred_spatemb, target_extra['similarity_raw_mask'].cuda(),
                                         labels=None, iou=True, iou_meter=iou_all_instances)
      iou_meter.update(iou_all_instances.avg)
      # loss_mask = 0
    elif "spatial_embedding" in args.losses:
      iou_all_instances = AverageMeter()
      pred_extra = F.interpolate(pred_extra, scale_factor=(1, 8, 8), mode='trilinear')
      # spatial embedding loss expents the last channel to be a seed map, which could be the fg/bg prediction here
      # pred_spatemb = torch.cat((pred_extra.cuda(), pred_mask[:, -1:]), dim=1)
      pred_spatemb = pred_extra
      criterion_extra = SpatialEmbLoss(n_sigma=args.embedding_dim - 4)
      loss_extra = criterion_extra(pred_spatemb, target_extra['similarity_raw_mask'].cuda(),
                                         labels=None, iou=True, iou_meter=iou_all_instances)
      iou_meter.update(iou_all_instances.avg)
      # loss_mask = 0

  # print("loss_extra {}".format(loss_extra))
  loss = loss_mask + loss_extra

  return loss, loss_image, pred_mask, loss_extra
