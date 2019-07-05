import torch


def bootstrapped_ce_loss(raw_ce, n_valid_pixels_per_im=None, fraction=0.25):
  n_valid_pixels_per_im = raw_ce.shape[-1]*raw_ce.shape[-2] if n_valid_pixels_per_im is None else n_valid_pixels_per_im
  ks = torch.max(torch.tensor(n_valid_pixels_per_im * fraction).cuda().int(), torch.tensor(1).cuda().int())
  bootstrapped_loss = raw_ce.reshape(raw_ce.shape[0], -1).topk(ks, dim=-1)[0].mean(dim=-1).mean()
  return bootstrapped_loss