from collections import defaultdict
from threading import Timer

from dask.tests.test_config import yaml
from pycocotools import mask as masktools

import math
import cv2
import json
import numpy as np
import os
import torch
import torch.nn.functional as F

from inference_handlers.infer_utils.Visualisation import create_color_map


def compute_scaled_padding(image_dims, min_dim, max_dim, mask_scale):
    lower_size = float(min(image_dims))
    higher_size = float(max(image_dims))

    scale_factor = min_dim / lower_size
    if (higher_size * scale_factor) > max_dim:
        scale_factor = max_dim / higher_size

    height, width = image_dims
    resized_height, resized_width = round(scale_factor * height), round(scale_factor * width)

    padding_right, padding_bottom = (math.ceil(resized_width / 32.) * 32.) - resized_width, \
                                    (math.ceil(resized_height / 32.) * 32.) - resized_height
    return int(round(padding_bottom / mask_scale)), int(round(padding_right / mask_scale))


class YoutubeVISOutputGenerator(object):
    def __init__(self, output_dir, outlier_label, save_visualization, category_mapping_file, category_names_file,
                 *args, **kwargs):

        with open(category_mapping_file, 'r') as fh:
            category_mapping = yaml.load(fh, Loader=yaml.SafeLoader)

        # -----debug----
        # with open('/home/athar/code_repos/masktcnn/masktcnn/data/metainfo/youtube_vis_mapped_category_names.yaml', 'r') as fh:
        #     self.mapped_cat_names = yaml.load(fh, Loader=yaml.SafeLoader)

        # category_mapping contains a mapping from the original youtube vis class labels to the merged labels that were
        # used to train the model. This mapping now has to be reversed to obtain the original class labels for uploading
        # results.
        self.category_mapping = defaultdict(list)
        for orig_label, mapped_label in category_mapping.items():
            self.category_mapping[mapped_label].append(orig_label)  # one to many mapping is possible

        self.outlier_label = outlier_label
        self.instances = []

        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.save_visualization = save_visualization

        self.category_names = {}
        if save_visualization:
            with open(category_names_file, 'r') as fh:
                self.category_names = yaml.load(fh, Loader=yaml.SafeLoader)

    @staticmethod
    def compute_instance_confidences(instance_pt_counts, instance_ids_to_keep):
        # set instance confidence based on number of points in the instance mask across the entire sequence.
        instance_pt_counts = {
            instance_id: count
            for instance_id, count in instance_pt_counts.items() if instance_id in instance_ids_to_keep
        }

        max_pts = float(max(list(instance_pt_counts.values())))
        return {
            instance_id: float(count) / max_pts for instance_id, count in instance_pt_counts.items()
        }

    def process_sequence(self, track_mask_idxes, track_mask_labels, instance_pt_counts, category_masks, mask_dims,
                         max_tracks, device="cpu"):
        """
        Given a list of mask indices per frame, creates a sequence of masks for the entire sequence.
        :param sequence: instance of YoutubeVISSequence
        :param track_mask_idxes: list(tuple(tensor, tensor))
        :param track_mask_labels: list(tensor)
        :param instance_pt_counts: dict(int -> int)
        :param category_masks: tensor(T, H, W) of type long
        :param mask_dims: tuple(int, int) (height, width)
        :param mask_scale: int
        :param max_tracks: int
        :param device: str
        :return: None
        """
        mask_height, mask_width = mask_dims
        assert len(track_mask_idxes) == len(track_mask_labels)
        assert category_masks.shape[0] == len(track_mask_idxes)
        assert tuple(category_masks.shape[-2:]) == tuple(mask_dims), \
            "Shape mismatch between semantic masks {} and embedding masks {}".format(category_masks.shape, mask_dims)
        assert max_tracks < 256

        instances_to_keep = track_mask_labels[:max_tracks]
        # num_tracks = len(instances_to_keep)
        print("Instances to keep: ", instances_to_keep)

        instance_confidences = self.compute_instance_confidences(instance_pt_counts, instances_to_keep)
        instance_semantic_label_votes = defaultdict(lambda: {k: 0 for k in self.category_mapping.keys()})
        instance_rle_masks = {k: [] for k in instances_to_keep}

        for t in range(len(track_mask_idxes)):
            # filter semantic labels for background pixels
            category_mask_t = category_masks[t][track_mask_idxes[t]]

            assert category_mask_t.shape == track_mask_labels[t].shape, \
                "Shape mismatch between category labels {} and instance labels {}".format(
                    category_mask_t.shape, track_mask_labels[t].shape)

            mask_t = []
            for instance_id in instances_to_keep:
                label_mask = track_mask_labels[t] == instance_id
                mask = torch.zeros(mask_height, mask_width, dtype=torch.long, device=device)
                mask[track_mask_idxes[t]] = label_mask.long()
                mask_t.append(mask)

            # count votes for the semantic label of each instance
            for i, instance_id in enumerate(instances_to_keep):
                active_semantic_labels, label_counts = \
                    category_mask_t[track_mask_labels[t] == instance_id].unique(return_counts=True)

                active_semantic_labels = active_semantic_labels.tolist()
                label_counts = label_counts.tolist()

                for label, count in zip(active_semantic_labels, label_counts):
                    if label != 0:
                        instance_semantic_label_votes[instance_id][label] += count

            mask_t = torch.stack(mask_t, dim=0)
            mask_t = (mask_t > 0.5).byte().squeeze(0)  # [N, H, W]

            for i, instance_id in enumerate(instances_to_keep):
                rle_mask = masktools.encode(np.asfortranarray(mask_t[i].numpy()))
                rle_mask["counts"] = rle_mask["counts"].decode("utf-8")  # bytes to utf-8 so that json.dump works
                instance_rle_masks[instance_id].append(rle_mask)

        self.add_sequence_result(sequence, instance_rle_masks, instance_semantic_label_votes, instance_confidences)

    def add_sequence_result(self, seq, instance_rle_masks, instance_semantic_label_votes, instance_confidences):
        # assign semantic label to each instance based on max votes
        instance_mapped_labels = dict()
        for instance_id in instance_rle_masks:
            semantic_label_votes = instance_semantic_label_votes[instance_id]
            max_voted_label, num_votes = max([
                (semantic_label, votes) for semantic_label, votes in semantic_label_votes.items()], key=lambda x: x[1])

            # map the predicted semantic label to the original labels in the Youtube VIS dataset spec
            assert max_voted_label in self.category_mapping, "Label {} does not exist in mapping".format(max_voted_label)
            instance_mapped_labels[instance_id] = self.category_mapping[max_voted_label]
            # print("Mapping {}({}) to {}".format(max_voted_label, self.mapped_cat_names[max_voted_label], instance_mapped_labels[instance_id]))

        sequence_instances = []
        for instance_id in instance_rle_masks:
            assert instance_id in instance_confidences, \
                "Instance ID {} has no associated confidence score".format(instance_id)

            for mapped_label in instance_mapped_labels[instance_id]:
                instance_dict = {
                    "video_id": seq.seq_id,
                    "score": instance_confidences[instance_id],
                    "category_id": mapped_label,
                    "segmentations": instance_rle_masks[instance_id]
                }
                sequence_instances.append(instance_dict)

        if self.save_visualization:
            self.save_sequence_visualizations(seq, sequence_instances)

        self.instances.extend(sequence_instances)

    def save_sequence_visualizations(self, seq, instances):
        cmap = create_color_map().tolist()
        seq_output_dir = os.path.join(self.output_dir, 'vis', str(seq.seq_id))
        os.makedirs(seq_output_dir, exist_ok=True)

        with seq:
            images = seq.images

        for t, image_t in enumerate(images):
            for n, instance in enumerate(instances, 1):
                segmentations = instance["segmentations"]
                assert len(segmentations) == len(images)

                category_label = self.category_names[instance['category_id']]
                color = cmap[n]

                segmentation_t = segmentations[t].copy()
                segmentation_t["counts"] = segmentation_t["counts"].encode("utf-8")

                mask = masktools.decode(segmentation_t)

                annotation_text = "{} {:02f}".format(category_label, instance["score"])
                image_t = annotate_instance(image_t, mask, color, annotation_text)

            cv2.imwrite(os.path.join(seq_output_dir, '{:05d}.jpg'.format(t)), image_t)

    def save(self, *args, **kwargs):
        """
        Writes out results to disk
        :return: None
        """
        with open(os.path.join(self.output_dir, 'results.json'), 'w') as fh:
            json.dump(self.instances, fh)

