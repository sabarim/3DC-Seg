import cv2
import numpy as np
import torch
from PIL import Image

from utils import cv2_util
from utils.util import get_iou


def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

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


def save_mask(mask, img_path):
    if np.max(mask) > 255:
        raise ValueError('Maximum id pixel value is 255')
    mask_img = Image.fromarray(mask.astype(np.uint8))
    mask_img.putpalette(color_map().flatten().tolist())
    mask_img.save(img_path)


def compute_colors_for_labels(labels):
    """
    Simple function that adds fixed colors depending on the class
    """
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = labels[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")
    return colors


def select_top_predictions(predictions, confidence_threshold):
    """
    Select only predictions which have a `score` > self.confidence_threshold,
    and returns the predictions in descending order of score

    Arguments:
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `scores`.

    Returns:
        prediction (BoxList): the detected objects. Additional information
            of the detection properties can be found in the fields of
            the BoxList via `prediction.fields()`
    """
    scores = predictions.get_field("scores")
    keep = torch.nonzero(scores > confidence_threshold).squeeze(1)
    predictions = predictions[keep]
    scores = predictions.get_field("scores")
    _, idx = scores.sort(0, descending=True)
    return predictions[idx]


def overlay_boxes(image, predictions):
    """
    Adds the predicted boxes on top of the image

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `labels`.
    """
    labels = predictions.get_field("labels")
    boxes = predictions.bbox

    colors = compute_colors_for_labels(labels).tolist()

    for box, color in zip(boxes, colors):
        box = box.to(torch.int64)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        image = cv2.rectangle(
            image, tuple(top_left), tuple(bottom_right), tuple(color), 1
        )

    return image


def overlay_mask(image, predictions):
    """
    Adds the instances contours for each predicted object.
    Each label has a different color.

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `mask` and `labels`.
    """
    masks = predictions.get_field("mask").numpy()
    labels = predictions.get_field("labels")

    colors = compute_colors_for_labels(labels).tolist()

    for mask, color in zip(masks, colors):
        thresh = mask[0, :, :, None]
        contours, hierarchy = cv2_util.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        image = cv2.drawContours(image, contours, -1, color, 3)

    composite = image

    return composite


def overlay_class_names(image, predictions):
    """
    Taken from the maskrcnn_benchmark code
    Adds detected class names and scores in the positions defined by the
    top-left corner of the predicted bounding box

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `scores` and `labels`.
    """
    scores = predictions.get_field("scores").tolist()
    labels = predictions.get_field("labels").tolist()
    # labels = [self.CATEGORIES[i] for i in labels]
    boxes = predictions.bbox

    template = "{}: {:.2f}"
    for box, score, label in zip(boxes, scores, labels):
        x, y = box[:2]
        s = template.format(label, score)
        cv2.putText(
            image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
        )

    return image


def overlay_predicitons(image, predictions):
    """
    Taken from the maskrcnn_benchmark code
    Arguments:
        image (np.ndarray): an image as returned by OpenCV

    Returns:
        prediction (BoxList): the detected objects. Additional information
            of the detection properties can be found in the fields of
            the BoxList via `prediction.fields()`
    """
    top_predictions = select_top_predictions(predictions, confidence_threshold=0.7)

    result = image.copy()
    result = overlay_boxes(result, top_predictions)
    result = overlay_mask(result, top_predictions)
    result = overlay_class_names(result, top_predictions)

    return result


def write_output_mask(proposals, path):
    result_mask = np.zeros(list(proposals['mask'].shape[2:]))
    for i in range(len(proposals['mask'])):
        result_mask[proposals['mask'][i,0].data.cpu().numpy() == 1] = proposals['track_ids'][i] + 1
    save_mask(result_mask, path)


def get_one_hot_vectors(mask, num_objects = None):
    num_objects = np.max(mask) if num_objects is None else num_objects
    one_hot_mask = np.zeros((num_objects, ) + mask.shape)
    for i in range(num_objects):
        one_hot_mask[i] = (mask == i).astype(np.uint8)

    return one_hot_mask


def top_n_predictions_maskrcnn(predictions, n):
    """
    Select top n predictions based on score

    Arguments:
        predictions (dictionary): the result of the computation by the model.
            It should contain the field `scores`.

    Returns:
        prediction (dictionary): the detected objects. Additional information
            of the detection properties can be found in the fields of
            the BoxList via `prediction.fields()`
    """
    scores = predictions["scores"]
    n = min(n, len(scores))
    keep = torch.zeros(scores.shape).bool()
    topk, indices = torch.topk(scores, n)
    keep[indices] = True
    predictions = {key:predictions[key][keep] for key in predictions.keys()}
    # scores = predictions["scores"]
    # _, idx = scores.sort(0, descending=True)
    return predictions


def filter_by_category(predictions, filter_cats):
    """
    Filter proposals by category

    Arguments:
        predictions (dictionary): the result of the computation by the model.
            It should contain the field 'labels'.

    Returns:
        prediction (dictionary): the detected objects. Additional information
            of the detection properties can be found in the fields of
            the BoxList via `prediction.fields()`
    """
    cats = predictions["labels"]
    keep = np.zeros_like(cats).astype(np.bool)
    filtered_predictions = {}
    for cat in filter_cats:
        keep[(cats == cat).data.cpu().numpy()] = True
    filtered_predictions.update({key: predictions[key][keep] for key in predictions.keys()})
    return filtered_predictions


def create_object_id_mapping(ref_mask, proposals):
  ids = np.setdiff1d(np.unique(ref_mask), [0])
  result = {}
  for id in ids:
    _, target_id = get_best_overlap((ref_mask == id).astype(np.uint8), proposals)
    result[id] = target_id

  return result


def get_best_overlap(ref_obj, proposals):
    best_iou = 0
    target_id = -1
    # mask = proposals[:, 0].cuda()

    for obj_id in range(len(proposals)):
        iou = get_iou(ref_obj, proposals[obj_id].astype(np.uint8))
        if iou > best_iou and iou > 0.1:
            best_iou = iou
            target_id = obj_id
            # mask = (proposals[:, 0] == obj_id).int().cuda()

    return best_iou, target_id