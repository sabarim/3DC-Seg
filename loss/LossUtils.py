import torch
from functools import reduce


def parse_embedding_output(model_embedding_output, embedding_size):
  """
  Converts the raw model output (inverse variances + correlation coefficients) to values corresponding to the
  precision matrix (inverse covariance matrix).

  References:
  (1) https://en.wikipedia.org/wiki/Partial_correlation#Using_matrix_inversion
  (2) https://stats.stackexchange.com/questions/121979/get-correlation-matrix-of-3-variables-from-any-combination-of-3-simples-partials

  :param model_embedding_output: tensor(N1, ...., Nn, E)
  :return: tensor(N1, ...., Nn, E)
  """
  # number of non-diagonal entries for a square matrix = (n^2 - n) / 2
  num_nondiag_vals = int((embedding_size ** 2 - embedding_size) / 2.)

  # ensure correct dimension sizes: last dimension should have n entries corresponding to the diagonal values of the
  # precision matrix and (n^2 - n) / 2 values correpsonding to the non-diagonal correlations.
  assert model_embedding_output.shape[-1] == num_nondiag_vals + embedding_size, \
    "For embedding size {}, model output should have size {} at last dim, but got {}".format(
      embedding_size, num_nondiag_vals + embedding_size, model_embedding_output.shape[-1])

  if model_embedding_output.ndimension() > 1:
    # Flatten the tensor: [N1, ...., Nn, E] --> [N1*....*Nn, E]
    remaining_dims = tuple(model_embedding_output.shape[:-1])
    remaining_numel = reduce(lambda x, y: x * y, tuple(model_embedding_output.shape[:-1]))
    model_embedding_output = model_embedding_output.view(remaining_numel, model_embedding_output.shape[-1])
  else:
    remaining_dims = None

  # convention: first set of n values represent inverse variances, and the remaining (n^2 - n) / 2 values after that
  # represent the correlation coefficients in the same order as returned by torch.triu_indices.
  inverse_vars, corr_coeffs = model_embedding_output.split((embedding_size, num_nondiag_vals), dim=1)

  # activation function: exp() for inverse variances, tanh() for correlation coefficients
  inverse_vars = (inverse_vars * 10).exp()
  corr_coeffs = (corr_coeffs * 0.5).tanh()

  # convert correlation coefficients to non-diagonal entries of the precision matrix
  nondiag_idxes = torch.triu_indices(embedding_size, embedding_size, 1).t().tolist()  # [(n^2 - n) / 2, 2]
  precision_coeffs = []

  for i, (dim1, dim2) in enumerate(nondiag_idxes):
    corr_12 = corr_coeffs[:, i]
    inv_var1, inv_var2 = inverse_vars[:, dim1], inverse_vars[:, dim2]
    precision_coeffs.append(-1. * corr_12 * (inv_var1 * inv_var2).clamp(min=1e-8).sqrt())

  precision_coeffs = torch.stack(precision_coeffs, 1)
  parsed_output = torch.cat((inverse_vars, precision_coeffs), 1)

  # restore original tensor dimensions
  if remaining_dims:
    parsed_output = parsed_output.view(*remaining_dims, parsed_output.shape[1])

  return parsed_output


def precision_tensor_to_matrix(precision, embedding_size, compute_mean = True):
  """
  Converts a set of values for the precision matrix to a matrix after averaging.
  :param precision: tensor(N1, ..., Nn, E). This should be the output of the model after applying
  'parse_embedding_output'.
  :param embedding_size: int
  :return: tensor(N, E, E)
  """
  precision_mat = [[None for _ in range(embedding_size)] for _ in range(embedding_size)]
  shape = (1, embedding_size, embedding_size) if compute_mean else (precision.shape[0], embedding_size, embedding_size)
  precision_mat = torch.zeros(shape, dtype=precision.dtype,
                              device=precision.device)

  # number of non-diagonal entries for a square matrix = (n^2 - n) / 2
  num_nondiag_vals = int((embedding_size ** 2 - embedding_size) / 2.)

  assert precision.shape[-1] == num_nondiag_vals + embedding_size, \
    "For embedding size {}, model output should have size {} at last dim, but got {}".format(
      embedding_size, num_nondiag_vals + embedding_size, precision.shape[-1])

  # average the values of inverse vars/covars
  if precision.ndimension() > 1:
    precision = precision.view(-1, precision.shape[-1]).mean(0).unsqueeze(0) if compute_mean else\
      precision.view(-1, precision.shape[-1])

  # populate diagonal entries
  for i in range(embedding_size):
    precision_mat[:, i,i] = precision[:, i]

  # populate non-diagonal entries
  nondiag_idxes = torch.triu_indices(embedding_size, embedding_size, 1).t().tolist()  # [num_nondiag_vals, 2]
  for i in range(len(nondiag_idxes)):
    dim1, dim2 = nondiag_idxes[i]
    prec_12 = precision[:, i]
    precision_mat[:, dim1,dim2] = prec_12
    precision_mat[:, dim2, dim1] = prec_12

  # return torch.stack([torch.stack(col) for col in precision_mat])
  return precision_mat.squeeze()


def mahalanobis_distance(x, center, precision_mat, return_squared_distance=False):
  """
  Computes the Mahalanobis distance between a set of points x from some center
  :param x: tensor(N, E) (E = embedding size)
  :param center: tensor(E) or tensor(1, E)
  :param precision_mat: precision matrix as tensor(E, E)
  :param return_squared_distance: if True, returns the squared distance (as in formula for multivariate gaussian)
  :return: tensor(N)
  """
  N, E = x.shape
  if center.ndimension() == 1:
    center = center.unsqueeze(0)  # [E] -> [1, E]

  x = x - center  # [N, E]
  x_t = x.unsqueeze(1)  # [N, 1, E]
  x = x.unsqueeze(2)  # [N, E, 1]
  if len(precision_mat.shape) == 2:
    precision_mat = precision_mat.unsqueeze(0).expand(N, -1, -1)  # [N, E, E]

  dists = torch.bmm(x_t, precision_mat)  # [N, 1, E] * [N, E, E] = [N, 1, E]
  dists = torch.bmm(dists, x)  # [N, 1, E] * [N, E, 1] = [N, 1, 1]
  dists = dists.squeeze(0).squeeze(0)
  if return_squared_distance:
    return dists
  else:
    return dists.clamp(min=1e-8).sqrt()