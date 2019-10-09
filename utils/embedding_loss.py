import torch
from torch.nn import functional as F

def smooth_l1_loss(loss):
  return torch.where(loss < 1., torch.pow(loss, 2), loss)

def make_zero_tensor(dtype, device, required_grad):
  return torch.tensor(0, dtype=dtype, device=device, requires_grad=required_grad)


def compute_embedding_loss(embedding_map, targets):
  """
  Computes the embedding loss.
  :param embedding_map: Tensor of shape [N, E (embedding dimensionality), T, H, W]
  :param targets: List (length N) of dicts, each containing a 'masks' field containing a tensor of 
  shape (I (instances), T, H, W)
  :return: Tuple of 3 losses (only the first value has to be back-proped); the rest are for logging)
  """
  embedding_map = embedding_map.permute(0, 2, 3, 4, 1)  # [N, T, H, W, E]

  losses_variance = torch.zeros((0,), dtype=torch.float32, device=embedding_map.device)
  losses_distance = torch.zeros((0,), dtype=torch.float32, device=embedding_map.device)
  losses_regularization = torch.zeros((0,), dtype=torch.float32, device=embedding_map.device)

  DELTA_VAR = 0.1  # distance of instance pixels to their respective centers
  DELTA_DISTANCE = 0.5  # distance between centers of different instances (/2)

  W_VAR_LOSS = 1.0
  W_DISTANCE_LOSS = 1.0
  W_REGULARIZATION_LOSS = 1e-3

  # The masks have to be resized to match the size of the embedding map
  SCALE_FACTOR = 0.125

  for embeddings_per_seq, targets_per_seq in zip(embedding_map, targets):
      masks = targets_per_seq
      # downscale masks to match size of emebedding map
      masks = F.interpolate(
          masks.float(), scale_factor=SCALE_FACTOR, mode='nearest').to(torch.uint8)

      assert masks.shape[-2:] == embeddings_per_seq.shape[1:3], \
          "Masks tensor has shape {} while embedding map has shape {}".format(masks.shape, embeddings_per_seq.shape)

      nonzero_mask_pts = masks.nonzero()
      if nonzero_mask_pts.shape[0] == 0:
          print("[ WARN] No valid mask points exist in sample.")
          continue

      _, instance_pt_counts = nonzero_mask_pts[:, 0].unique(sorted=True, return_counts=True)
      instance_id_sort_idx = nonzero_mask_pts[:, 0].argsort()
      nonzero_mask_pts = nonzero_mask_pts[instance_id_sort_idx]
      nonzero_mask_pts = nonzero_mask_pts.split(tuple(instance_pt_counts.tolist()))
      nonzero_mask_pts = tuple([nonzero_mask_pts[i].unbind(dim=1)[1:] for i in range(len(nonzero_mask_pts))])

      instance_embeddings = [
          embeddings_per_seq[nonzero_mask_pts[i]]
          for i in range(len(nonzero_mask_pts))
      ]  # list(tensor[M, 8])

      instance_embedding_means = [emb.mean(dim=0, keepdim=True) for emb in instance_embeddings]

      instance_embedding_diffs = [
          smooth_l1_loss(((torch.pow(emb_mean - embs, 2).sum(1) + 1e-8).sqrt() - DELTA_VAR).clamp(min=0)).mean()
          for (emb_mean, embs) in zip(instance_embedding_means, instance_embeddings)
      ]
      losses_variance = torch.cat((losses_variance, torch.stack(instance_embedding_diffs)))

      # compute repulsive mean between mean embedding of different clusters
      if instance_pt_counts.numel() > 1:
          instance_embedding_means = torch.stack(instance_embedding_means).squeeze(1)  # [I, E]
          distance_loss = smooth_l1_loss(((2. * DELTA_DISTANCE) - F.pdist(instance_embedding_means)).clamp(min=0))
          losses_distance = torch.cat((losses_distance, distance_loss))

      # compute regularization loss
      instance_embeddings = torch.cat(instance_embeddings, dim=0)
      losses_regularization = torch.cat((losses_regularization, instance_embeddings.norm(dim=-1)))

  if losses_variance.numel() == 0:
      losses_distance = make_zero_tensor(torch.float32, embedding_map.device, True)
      losses_variance = make_zero_tensor(torch.float32, embedding_map.device, True)
      losses_regularization = make_zero_tensor(torch.float32, embedding_map.device, True)
  else:
      losses_variance = losses_variance.mean() * W_VAR_LOSS
      losses_regularization = losses_regularization.mean() * W_REGULARIZATION_LOSS
      if losses_distance.numel() == 0:
          losses_distance = make_zero_tensor(torch.float32, embedding_map.device, True)
      else:
          losses_distance = losses_distance.mean() * W_DISTANCE_LOSS

  total_loss = losses_variance + losses_distance + losses_regularization

  return total_loss, losses_variance, losses_distance
