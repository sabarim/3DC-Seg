import cv2
import torch
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse, Patch


def create_color_map(N=256, normalized=False):
  def bitget(byteval, idx):
    return (byteval & (1 << idx)) != 0

  dtype = 'float32' if normalized else 'uint8'
  cmap = np.zeros((N, 3), dtype=dtype)
  for i in range(N):
    r = g = b = 0
    c = i
    for j in range(8):
      r = r | (bitget(c, 0) << 7 - j)
      g = g | (bitget(c, 1) << 7 - j)
      b = b | (bitget(c, 2) << 7 - j)
      c = c >> 3

    cmap[i] = np.array([r, g, b])

  cmap = cmap / 255 if normalized else cmap
  return cmap

def overlay_mask_on_image(image, mask, mask_opacity=0.6, mask_color=(0, 255, 0)):
  if mask.ndim == 3:
    assert mask.shape[2] == 1
    _mask = mask.squeeze(axis=2)
  else:
    _mask = mask
  mask_bgr = np.stack((_mask, _mask, _mask), axis=2)
  masked_image = np.where(mask_bgr > 0, mask_color, image)
  return ((mask_opacity * masked_image) + ((1. - mask_opacity) * image)).astype(np.uint8)


def bbox_from_mask(mask, order='Y1Y2X1X2', return_none_if_invalid=False):
  reduced_y = np.any(mask, axis=0)
  reduced_x = np.any(mask, axis=1)

  x_min = reduced_y.argmax()
  if x_min == 0 and reduced_y[0] == 0:  # mask is all zeros
    if return_none_if_invalid:
      return None
    else:
      return -1, -1, -1, -1

  x_max = len(reduced_y) - np.flip(reduced_y, 0).argmax()

  y_min = reduced_x.argmax()
  y_max = len(reduced_x) - np.flip(reduced_x, 0).argmax()

  if order == 'Y1Y2X1X2':
    return y_min, y_max, x_min, x_max
  elif order == 'X1X2Y1Y2':
    return x_min, x_max, y_min, y_max
  elif order == 'X1Y1X2Y2':
    return x_min, y_min, x_max, y_max
  elif order == 'Y1X1Y2X2':
    return y_min, x_min, y_max, x_max
  else:
    raise ValueError("Invalid order argument: %s" % order)


def annotate_instance(image, mask, color, text_label):
    """
    :param image: np.ndarray(H, W, 3)
    :param mask: np.ndarray(H, W)
    :param color: tuple/list(int, int, int) in range [0, 255]
    :param text_label: str
    :return: np.ndarray(H, W, 3)
    """
    assert image.shape[:2] == mask.shape, "Shape mismatch between image {} and mask {}".format(image.shape, mask.shape)
    color = tuple(color)

    overlayed_image = overlay_mask_on_image(image, mask, mask_color=color)
    bbox = bbox_from_mask(mask, order='X1Y1X2Y2', return_none_if_invalid=True)
    if not bbox:
      return overlayed_image

    xmin, ymin, xmax, ymax = bbox
    cv2.rectangle(overlayed_image, (xmin, ymin), (xmax, ymax), color=color, thickness=2)
    # print(overlayed_image.shape)
    # print("Label: ", text_label)
    # print((xmin, ymin))
    cv2.putText(overlayed_image, text_label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

    return overlayed_image


def visualize_embeddings(embeddings, labels, bandwidths, plot_bandwidths, seediness=None, labels_start=1,
                         axes_labels=('T', 'Y', 'X')):
  """
  :param embeddings: list(T, tensor(M, E)) or tensor(M, E)
  :param labels: list(T, tensor(M)) or tensor(M)
  :param bandwidths: list(T, tensor(M, E)) or tensor(M, E)
  :param plot_bandwidths: bool
  :param seediness: list(T, tensor(M(,1))) or tensor(M(,1))
  :param labels_start: int
  :param axes_labels: list/tuple
  """
  # flatten embeddings and labels
  if isinstance(embeddings, (list, tuple)):
    embeddings = torch.cat(embeddings)
  if isinstance(labels, (list, tuple)):
    labels = torch.cat(labels).to(torch.int32)
  if isinstance(seediness, (list, tuple)):
    seediness = torch.cat(seediness)

  embedding_size = embeddings.shape[1]
  assert embeddings.shape[0] == labels.shape[0], \
    "Embeddings tensor has shape {} while labels tensor has shape {}".format(embeddings.shape, labels.shape)

  if torch.is_tensor(seediness):
    if seediness.ndimension() == 2:
      assert seediness.shape[1] == 1
      seediness = seediness.squeeze(1)
    assert seediness.shape[0] == embeddings.shape[0], \
      "Embeddings tensor has shape {} while seediness tensor has shape {}".format(embeddings.shape, seediness.shape)

  mean_embeddings = []
  mean_stds = []
  mean_labels = []

  if plot_bandwidths:  # visualize bandwidths of the predicted embeddings
    if isinstance(bandwidths, (list, tuple)):
      bandwidths = torch.cat(bandwidths)
    assert bandwidths.shape == embeddings.shape, \
      "Embeddings tensor has shape {} while bandwidths tensor has shape {}".format(embeddings.shape, bandwidths.shape)

    # split embeddings and bandwidths by label
    unique_labels = set(labels.unique().tolist()) - {-1}  # ignore outliers

    for l in unique_labels:
      mean_labels.append(l)
      mean_embeddings.append(embeddings[labels == l].mean(dim=0))
      mean_stds.append((1. / bandwidths[labels == l].mean(dim=0).clamp(min=1e-8)).sqrt())

  embeddings = embeddings.numpy()
  axes_idx_pairs = list(zip(*np.triu_indices(embedding_size, 1)))
  assert len(axes_labels) == embedding_size

  cmap = create_color_map(normalized=True).tolist()
  colors = np.array([cmap[(l - labels_start + 1) % 256] if l != -1 else cmap[0] for l in labels.tolist()])

  fig = Figure(figsize=(10, 8), dpi=300)
  canvas = FigureCanvasAgg(fig)

  assert len(axes_idx_pairs) <= 3
  nrows = 2 if len(axes_idx_pairs) > 2 else 1
  ncols = 2
  axes = fig.subplots(nrows, ncols)

  if torch.is_tensor(seediness):
    fig_seediness = Figure(figsize=(10, 8), dpi=300)
    canvas_seediness = FigureCanvasAgg(fig_seediness)
    axes_seediness = fig_seediness.subplots(nrows, ncols)

  for i, (ax_idx1, ax_idx2) in enumerate(axes_idx_pairs):
    x, y = embeddings[:, ax_idx1], embeddings[:, ax_idx2]
    if x.size == 0:
      continue

    axis_min, axis_max = min(-1, x.min(), y.min()), max(1, x.max(), y.max())
    # y_min, y_max = min(-1, y.min()), max(1, y.max())

    ax = axes[i // nrows, i % nrows]
    ax.set_xlim([axis_min, axis_max])
    ax.set_ylim([axis_min, axis_max])

    ax.scatter(x, y, c=colors, marker='.', s=0.5, alpha=0.25)

    ax.set_title("{} / {}".format(axes_labels[ax_idx2], axes_labels[ax_idx1]))

    if torch.is_tensor(seediness):
      ax_s = axes_seediness[i // nrows, i % nrows]
      ax_s.set_xlim([axis_min, axis_max])
      ax_s.set_ylim([axis_min, axis_max])

      ax_s.scatter(x, y, c=(1. - seediness).tolist(), cmap='coolwarm', marker='.', s=0.8, alpha=1.0)
      # print(seediness.min(), seediness.max(), seediness.mean(), (seediness < 0.3).sum().item(), (seediness > 0.7).sum().item())
      ax_s.set_title("{} / {}".format(axes_labels[ax_idx2], axes_labels[ax_idx1]))

    legend_artists, legend_labels = [], []
    for mean_instance_embedding, mean_instance_std, instance_label in zip(
        mean_embeddings, mean_stds, mean_labels):
      # print("Mean instance STD: ", mean_instance_std)
      x, y = mean_instance_embedding[ax_idx1], mean_instance_embedding[ax_idx2]
      ax_idx1_std, ax_idx2_std = mean_instance_std[ax_idx1], mean_instance_std[ax_idx2]

      color = cmap[(instance_label - labels_start + 1) % 256]

      ax.add_patch(Ellipse(
        (x, y), ax_idx1_std * 2, ax_idx2_std * 2, 0, ec=color, fill=False, ls=':', lw=1, zorder=4))
      # ax.add_patch(Ellipse(
      #     (x, y), ax_idx1_std * 6, ax_idx2_std * 6, 0, ec=color, fill=False, ls=':', lw=1, zorder=4))

      legend_artists.append(Patch(color=color))
      legend_labels.append("{:.3f}/{:.3f}".format(ax_idx2_std, ax_idx1_std))

    if legend_artists:
      ax.legend(legend_artists, legend_labels, fontsize="x-small")

  canvas.draw()
  s, (width, height) = canvas.print_to_buffer()
  clustering_vis = np.frombuffer(s, np.uint8).reshape((height, width, 4))

  if torch.is_tensor(seediness):
    canvas_seediness.draw()
    s, (width, height) = canvas_seediness.print_to_buffer()
    seediness_vis = np.frombuffer(s, np.uint8).reshape((height, width, 4))
  else:
    seediness_vis = None

  return clustering_vis, seediness_vis