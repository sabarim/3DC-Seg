import glob
import json
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.mask import toBbox
import matplotlib
import argparse
import datetime
import time
from tabulate import tabulate
import re
import os
import scoring_fnc
from functools import partial
import mycolormap

IOU_THRESHOLD = 0.5


def load_gt(exclude_classes=(), ignored_sequences=(), prefix_dir_name='single_image_annotations',
            dist_thresh=1000.0, area_thresh=10*10):
    gt_jsons = glob.glob("%s/*/*.json"%prefix_dir_name)
    #gt_jsons = glob.glob("%s/*/postproc*.json"%prefix_dir_name)
    gt = {}
    for gt_json in gt_jsons:
        # Exclude from eval
        matching = [s for s in ignored_sequences if s in gt_json]
        if len(matching) > 0: continue
        anns = json.load(open(gt_json))
        bboxes = []
        for ann in anns:

            # # NEW: ignore farther-than-T-meters-away labels!
            # # -----------------------------------------------------
            # pos3d = ann["pos3d"]
            # z_coord = pos3d[2]
            # if z_coord is None or pos3d[2] > dist_thresh:
            #         continue
            # # -----------------------------------------------------

            if ann["category"] in exclude_classes:
                continue
            extr = ann["extreme_points"]
            assert len(extr) == 4
            x0 = min([c[0] for c in extr])
            y0 = min([c[1] for c in extr])
            x1 = max([c[0] for c in extr])
            y1 = max([c[1] for c in extr])

            # NEW: ignore really small labels
            # -----------------------------------------------------
            w = x1 - x0
            h = y1 - y0
            if w*h < area_thresh:
                    continue
            # -----------------------------------------------------

            bboxes.append((x0, y0, x1, y1))
        gt[gt_json] = bboxes
    n_boxes = sum([len(x) for x in gt.values()], 0)
    print("number of gt boxes", n_boxes)
    return gt, n_boxes


def load_gt_categories(exclude_classes=(), ignored_sequences=(), prefix_dir_name='single_image_annotations'):
    #gt_jsons = glob.glob("%s/*/postproc*.json"%prefix_dir_name)
    gt_jsons = glob.glob("%s/*/*.json"%prefix_dir_name)

    gt_cat = {}
    gt_cat_map = {}

    cat_idx = 0
    for gt_json in gt_jsons:

        # Exclude from eval
        matching = [s for s in ignored_sequences if s in gt_json]
        if len(matching) > 0: continue

        anns = json.load(open(gt_json))

        categories = []
        for ann in anns:
            cat_str = ann["category"]
            if cat_str in exclude_classes:
                continue
            categories.append(cat_str)

            if cat_str not in gt_cat_map:
                gt_cat_map[cat_str] = cat_idx
                cat_idx += 1

        gt_cat[gt_json] = categories
    n_boxes = sum([len(x) for x in gt_cat.values()], 0)
    print("number of gt boxes", n_boxes)
    return gt_cat, n_boxes, gt_cat_map


def get_image_paths(prefix_dir_name='oxford_fixed', ignored_sequences=()):
    gt_imgs = glob.glob("data/%s/*/images/left/postproc*.png" % prefix_dir_name)
    im_names = []
    for gt_img in gt_imgs:

        # Exclude from eval
        matching = [s for s in ignored_sequences if s in gt_img]
        if len(matching) > 0: continue

        im_names.append(gt_img)
    return im_names


def load_proposals(folder, gt, ignored_sequences=(), score_fnc=lambda prop: prop["score"], min_tr_len=1):
    proposals = {}
    embeddings = {}
    for filename in gt.keys():
        prop_filename = os.path.join(folder, "/".join(filename.split("/")[-2:])) #folder + "/".join(filename.split("/")[-2:])
        emb_filename = prop_filename.replace('.json', '.npz')

        # Exclude from eval
        matching = [s for s in ignored_sequences if s in filename]
        if len(matching) > 0:
            continue

        # Load proposals
        try:
            props = json.load(open(prop_filename))
        except ValueError:
            print ("Error loading json: %s"%prop_filename)
            quit()

        # If tracks, only keep those if min length
        #if "track_length" in props[0]:
        #    props = [x for x in props if int(x["track_length"]) >= min_tr_len]

        if props is None:
            continue

        props = sorted(props, key=score_fnc, reverse=True)

        # Load embeddings
        embs = None
        if os.path.isfile(emb_filename):
            embs = np.load(emb_filename)
            embs = embs['a']

        if "bbox" in props[0]:
            bboxes = [prop["bbox"] for prop in props]
        else:
            bboxes = [toBbox(prop["segmentation"]) for prop in props]
        # convert from [x0, y0, w, h] (?) to [x0, y0, x1, y1]
        bboxes = [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in bboxes]
        proposals[filename] = bboxes

        if embs is not None:
            embeddings[filename] = embs
    return proposals, embeddings


def calculate_ious(bboxes1, bboxes2):
    """
    :param bboxes1: Kx4 matrix, assume layout (x0, y0, x1, y1)
    :param bboxes2: Nx$ matrix, assume layout (x0, y0, x1, y1)
    :return: KxN matrix of IoUs
    """
    min_ = np.minimum(bboxes1[:, np.newaxis, :], bboxes2[np.newaxis, :, :])
    max_ = np.maximum(bboxes1[:, np.newaxis, :], bboxes2[np.newaxis, :, :])
    I = np.maximum(min_[..., 2] - max_[..., 0], 0) * np.maximum(min_[..., 3] - max_[..., 1], 0)
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
    U = area1[:, np.newaxis] + area2[np.newaxis, :] - I
    assert (U > 0).all()
    IOUs = I / U
    assert (IOUs >= 0).all()
    assert (IOUs <= 1).all()
    return IOUs


def evaluate_proposals(gt, props, n_max_proposals=1000):
    all_ious = [] # ious for all frames
    for img, img_gt in gt.items():
        if len(img_gt) == 0:
            continue
        img_props = props[img]
        gt_bboxes = np.array(img_gt)
        prop_bboxes = np.array(img_props)
        ious = calculate_ious(gt_bboxes, prop_bboxes)

        #pad to n_max_proposals
        ious_padded = np.zeros((ious.shape[0], n_max_proposals))
        ious_padded[:, :ious.shape[1]] = ious[:, :n_max_proposals]
        all_ious.append(ious_padded)
    all_ious = np.concatenate(all_ious)
    if IOU_THRESHOLD is None:
        iou_curve = [0.0 if n_max == 0 else all_ious[:, :n_max].max(axis=1).mean() for n_max in range(0, n_max_proposals + 1)]
    else:
        assert 0 <= IOU_THRESHOLD <= 1
        iou_curve = [0.0 if n_max == 0 else (all_ious[:, :n_max].max(axis=1) > IOU_THRESHOLD).mean() for n_max in
                     range(0, n_max_proposals + 1)]
    return iou_curve


def evaluate_folder(gt, folder, ignored_sequences=(), score_fnc=lambda prop: prop["score"], min_tr_len=1):
    props, embeddings = load_proposals(folder, gt,
                                       ignored_sequences=ignored_sequences,
                                       score_fnc=score_fnc,
                                       min_tr_len=min_tr_len)
    iou_curve = evaluate_proposals(gt, props)

    # Draw
    #method_name = folder.replace("/", "_")

    # draw_missed_targets(gt, props, method_name=method_name, ignored_sequences=ignored_sequences)

    # iou_25 = iou_curve[25]
    # iou_50 = iou_curve[50]
    # iou_100 = iou_curve[100]
    # iou_200 = iou_curve[200]
    # iou_400 = iou_curve[400]
    # end_iou = iou_curve[-1]

    iou_10 = iou_curve[10]
    iou_25 = iou_curve[25]
    iou_50 = iou_curve[50]
    iou_100 = iou_curve[100]
    iou_700 = iou_curve[700]
    end_iou = iou_curve[-1]

    method_name = os.path.basename(os.path.dirname(folder+"/"))

    print("%s: R25: %1.2f, R50: %1.2f, R100: %1.2f, R200: %1.2f, R400: %1.2f, R_total: %1.2f" %
          (method_name,
           iou_10,
           iou_25,
           iou_50,
           iou_100,
           iou_700,
           end_iou))

    return iou_curve


def title_to_filename(plot_title):
    filtered_title = re.sub("[\(\[].*?[\)\]]", "", plot_title) # Remove the content within the brackets
    filtered_title = filtered_title.replace("_", "").replace(" ", "").replace(",", "_")
    return filtered_title


def export_table(export_dict, plot_title, output_dir):
    # Plot everything specified via export_dict
    if output_dir is not None:
        #eval_points = [25, 50, 100, 200, 400, 1000]
        eval_points = [10, 25, 50, 100, 700]
        table = []
        for item in export_dict.items():
            curve = item[1]
            label = item[0].replace('.', '')
            table.append([label] + [curve[x] for x in eval_points])
        table.sort(key=lambda x: x[2], reverse=True)
        header = [str(x) for x in eval_points]

        tab_latex = tabulate(table, headers=header, tablefmt="latex", floatfmt=".2f")
        print ("------------------------")
        print ("Table: %s"%plot_title)
        print (tab_latex)

        with open(os.path.join(output_dir, title_to_filename(plot_title) + '.tex'), "w") as f:
            f.write(tab_latex)


def make_plot(export_dict, plot_title, x_vals, color_hack=None, line_hack=None):
    # Plot everything specified via export_dict
    plt.figure()
    lw = 4
    for idx, item in enumerate(export_dict.items()):
        curve_label = item[0].replace('.', '')
        if color_hack is None:
            plt.plot(x_vals[0:700], item[1][0:700], label=curve_label, linewidth=lw)
        else:
            # Get injected color
            if curve_label not in color_hack:
                raise ValueError("make_plot: Colormap for %s not specified!" % curve_label)

            # Get injected linestyle; if there is one
            linsty = '-'
            if '4D-GVT' in curve_label:
            	linsty='--'

            color = color_hack[curve_label]
            plt.plot(x_vals[0:700], item[1][0:700], label=curve_label, linewidth=lw, c=color, linestyle=linsty)

    ax = plt.gca()
    #ax.set_xticks(np.arange(0, 1, 0.1))
    ax.set_yticks(np.arange(0, 1.2, 0.2))
    ax.set_xticks(np.asarray([25, 100, 200, 300, 500, 700]))
    plt.xlabel("# proposals")
    plt.ylabel("Recall")
    ax.set_ylim([0.0, 1.0])
    plt.legend()
    plt.grid()
    plt.title(plot_title)


def export_figs(export_dict, plot_title, output_dir, x_vals):
    # Export figs, csv
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, title_to_filename(plot_title) + ".pdf"), bbox_inches='tight')

        # Save to csv
        np.savetxt(os.path.join(output_dir, 'num_objects.csv'), np.array(x_vals), delimiter=',', fmt='%d')
        for item in export_dict.items():
            np.savetxt(os.path.join(output_dir, item[0] + '.csv'), item[1], delimiter=',', fmt='%1.4f')

color_dct_hack_cvpr19 = {
    # Redish
    'mp-rcnn': 'firebrick',
    'mp-rcnn-3D': 'tab:orange',

    # Greenish
    'm-rcnn-agnostic': 'tab:green',
    'm-rcnn-original': 'tab:olive', #'darkslategrey',

    # Blue
    'mx-rcnn': 'tab:blue',

    # Purple
    'sharpmask': 'tab:green', # 'tab:purple',

    # Orange
    '4D-GVT-coselect': 'darkviolet',
    '4D-GVT-all': 'tab:red',

    'CAMOT': 'tab:green',
    'CAMOT-500': 'tab:green',
    '4D-GVT-coselect-600': 'tab:orange',

    # Orange
    'log_ratio_objectness': 'tab:purple',
    'log_ratio_motion': 'tab:orange',
    'log_ratio_mask_consistency': 'tab:green',

    # 'decayed_joint_prob_objectness': 'tab:brown',
    # 'decayed_joint_prob_motion': 'tab:orange',
    # 'decayed_joint_prob_mask_consistency': 'tab:purple',
}

def evaluate_all_folders_oxford(gt, plot_title, ignored_sequences=(), user_specified_result_dir=None, output_dir=None):

    print ("----------- Evaluate Oxford Recall -----------")

    # Export dict
    export_dict = {

    }

    if FLAGS.eval_scoring_fnc:
        # camot_for_scoring_eval = "single_image_proposals/camot_tracklets_no_track_suppression/"
        camot_for_scoring_eval = "single_image_proposals/4D-GVT-all"
        for fnc_name in scoring_fnc.score_fnc_dict.keys():
            score_fnc_name = fnc_name
            score_fnc__ = scoring_fnc.score_fnc_dict[score_fnc_name]

            export_dict[score_fnc_name] = evaluate_folder(gt, camot_for_scoring_eval,
                                                          ignored_sequences=ignored_sequences,
                                                          score_fnc=score_fnc__, min_tr_len=5)
    else:

        # +++ Pre-defined baselines +++
        if FLAGS.always_eval_baselines or user_specified_result_dir is None:
            mrcnn = evaluate_folder(gt, "single_image_proposals/MaskRCNN_original/",
                                    ignored_sequences=ignored_sequences)

            mrcnn_agnostic = evaluate_folder(gt, "single_image_proposals/MaskRCNN_agnostic_from_scratch/",
                                             ignored_sequences=ignored_sequences)

            mrcnn_two_heads_COCO = evaluate_folder(gt, "single_image_proposals/MaskRCNN_two_heads_COCO/",
                                                   ignored_sequences=ignored_sequences)

            #mrcnn_two_heads_semantic_suppression_iou = evaluate_folder(gt,
            #                                                           "single_image_proposals/MaskRCNN_two_heads_semantic_suppression_iou/",
            #                                                           ignored_sequences=ignored_sequences)

            mrcnn_two_heads_semantic_suppression_iou = evaluate_folder(gt,
                                                                       "single_image_proposals/MaskRCNN_two_heads_3D+semnms/",
                                                                       ignored_sequences=ignored_sequences)

            sharpmask = evaluate_folder(gt, "single_image_proposals/sharpmask/",
                                        ignored_sequences=ignored_sequences)

            camot_v1 = evaluate_folder(gt, "single_image_proposals/camot_fixed_gaiting/",
                                       ignored_sequences=ignored_sequences)
            export_dict = {
                "Mask R-CNN": mrcnn,
                "Mask R-CNN (category agnostic)": mrcnn_agnostic,
                "Mask R-CNN two-heads": mrcnn_two_heads_COCO,
                "Mask R-CNN, two-heads with 3D filter + semnms": mrcnn_two_heads_semantic_suppression_iou,
                "SharpMask": sharpmask,
                "CAMOT++, fixed-gaiting": camot_v1,
            }

        # +++ User-specified +++
        user_specified_results = None
        if user_specified_result_dir is not None:
            dirs = os.listdir(user_specified_result_dir)
            for dir in dirs:
                print ("---Eval: %s ---"%dir)
                user_specified_results = evaluate_folder(gt,
                                                         os.path.join(user_specified_result_dir, dir),
                                                         ignored_sequences=ignored_sequences)
                export_dict[dir] = user_specified_results



    # for beta in np.arange(0.1, 1.1, 0.2):
    #     export_dict["log_ratio_motion_and_mask_consistency" + str(beta)] = \
    #             evaluate_folder(gt, camot_for_scoring_eval,
    #                             ignored_sequences=ignored_sequences,
    #                             score_fnc=partial(scoring_fnc.sort_fnc_log_ratio_motion_and_mask_consistency_and_objectness,
    #                                               beta=beta),
    #                             min_tr_len=5)

    # for ms in [0.1, 0.2, 0.3, 0.5, 0.6]:
    #     export_dict["camot_len_ms_" + str(ms)] = \
    #         evaluate_folder(gt, camot_for_scoring_eval,
    #                         ignored_sequences=ignored_sequences,
    #                         score_fnc=partial(scoring_fnc.sort_fnc_log_ratio_objectness_normalized, min_score=ms, K=10),
    #                         min_tr_len=5)

    # for tr_len_filter in [1, 2, 3, 4, 5, 6, 7]:
    #     export_dict["camot_len_filt_" + str(tr_len_filter)] = \
    #         evaluate_folder(gt, "single_image_proposals/camot_fixed_gaiting/",
    #                         ignored_sequences=ignored_sequences, min_tr_len=tr_len_filter,
    #                         score_fnc=scoring_fnc.sort_fnc_norm_objectness_motion_mask)

    x_vals = range(1001)

    # Plot everything specified via export_dict
    make_plot(export_dict, plot_title, x_vals, color_hack=color_dct_hack_cvpr19)

    # Export figs, csv
    export_figs(export_dict, plot_title, output_dir, x_vals)

    # Export latex table(s)
    export_table(export_dict, plot_title, output_dir)


def evaluate_all_folders_kitti(gt, plot_title, ignored_sequences=(), user_specified_result_dir=None, output_dir=None):

    print ("----------- Evaluate KITTI Recall -----------")

    # Export dict
    export_dict = {

    }

    # +++ User-specified +++
    #user_specified_results = None
    if user_specified_result_dir is not None:
        dirs = os.listdir(user_specified_result_dir)
        for dir in dirs:
            user_specified_results = evaluate_folder(gt,
                                                     os.path.join(user_specified_result_dir, dir),
                                                     ignored_sequences=ignored_sequences)
            export_dict[dir] = user_specified_results
    # +++ Pre-defined baselines +++
    else:
        base_path = '/home/osep/projects/video-object-mining/data/kitti'
        props_dir_mrcnn = os.path.join(base_path, 'twoheads-700-kitti-tracking')
        props_dir_mrcnn_3D = os.path.join(base_path, 'twoheads-700-3D-kitti-tracking')
        props_dir_sm = os.path.join(base_path, 'sharpmask-1K')

        mrcnn_two_heads_COCO = evaluate_folder(gt, props_dir_mrcnn, ignored_sequences=ignored_sequences)
        mrcnn_two_heads_COCO_3D = evaluate_folder(gt, props_dir_mrcnn_3D, ignored_sequences=ignored_sequences)
        sharpmask = evaluate_folder(gt, props_dir_sm, ignored_sequences=ignored_sequences)

        export_dict = {
            #"Mask R-CNN": mrcnn,
            #"Mask R-CNN (category agnostic)": mrcnn_agnostic,
            "Mask R-CNN two-heads": mrcnn_two_heads_COCO,
            "Mask R-CNN two-heads, 3D filt.": props_dir_mrcnn_3D,
            #"Mask R-CNN, two-heads with semantic suppression": mrcnn_two_heads_semantic_suppression_iou,
            "Sharpmask": sharpmask,
        }

    x_vals = range(1001)

    # Plot everything specified via export_dict
    make_plot(export_dict, plot_title, x_vals, color_hack=color_dct_hack_cvpr19)

    # Export figs, csv
    export_figs(export_dict, plot_title, output_dir, x_vals)

    # Export latex table(s)
    export_table(export_dict, plot_title, output_dir)


def eval_recall_oxford(output_dir):

    # Experimental settings
    sequences_to_ignore = ['2015-11-06-11-21-12_sub_0032', '2015-11-06-11-21-12_sub_0017']
    #sequences_to_ignore = ['2015-03-10-14-18-10', '2014-08-08-13-01-14', '2014-07-14-14-49-50']
    #plt.figure(figsize=(8, 23))

    # Recall evaluation
    if FLAGS.evaluate_recall:
        print ("Evaluation: Recall")

        # +++ Most common categories +++
        print("evaluating car, bike, person, bus:")
        exclude_classes = ("other",)
        gt, n_gt_boxes = load_gt(exclude_classes, prefix_dir_name=FLAGS.labels, ignored_sequences=sequences_to_ignore)
        gt_categories, n_gt_boxes, category_map = load_gt_categories(exclude_classes,
                                                                     prefix_dir_name=FLAGS.labels,
                                                                     ignored_sequences=sequences_to_ignore)

        #plt.subplot(3, 1, 1)
        evaluate_all_folders_oxford(gt, "car, bike, person, and bus (" + str(n_gt_boxes) + " bounding boxes)",
                                    ignored_sequences=sequences_to_ignore, output_dir=output_dir,
                                    user_specified_result_dir=FLAGS.evaluate_dir)

        # +++ "other" categories +++
        print("evaluating others:")
        exclude_classes = ("car", "bike", "person", "bus")
        gt, n_gt_boxes = load_gt(exclude_classes, prefix_dir_name=FLAGS.labels, ignored_sequences=sequences_to_ignore)
        gt_categories, n_gt_boxes, category_map = load_gt_categories(exclude_classes,
                                                                     prefix_dir_name=FLAGS.labels,
                                                                     ignored_sequences=sequences_to_ignore)

        #plt.subplot(3, 1, 2)
        evaluate_all_folders_oxford(gt, "others (" + str(n_gt_boxes) + " bounding boxes)",
                                    ignored_sequences=sequences_to_ignore, output_dir=output_dir,
                                    user_specified_result_dir=FLAGS.evaluate_dir)

        # # +++ All categories +++
        # print("evaluating all:")
        # exclude_classes = ()
        # gt, n_gt_boxes = load_gt(exclude_classes, prefix_dir_name=FLAGS.labels, ignored_sequences=sequences_to_ignore)
        # gt_categories, n_gt_boxes, category_map = load_gt_categories(exclude_classes,
        #                                                              prefix_dir_name=FLAGS.labels,
        #                                                              ignored_sequences=sequences_to_ignore)
        # #plt.subplot(3, 1, 3)
        # evaluate_all_folders_oxford(gt, "all (" + str(n_gt_boxes) + " bounding boxes)",
        #                             ignored_sequences=sequences_to_ignore, output_dir=output_dir,
        #                             user_specified_result_dir=FLAGS.evaluate_dir)
        # #plt.show()


def eval_recall_kitti(output_dir):

    label_prefix = 'kitti_annotations'
    sequences_to_ignore = ()

    # KITTI labels: "car", "pedestrian", "person_sitting", "cyclist", "van", "truck", "tram", "misc"

    # Recall evaluation
    if FLAGS.evaluate_recall:

        print ("Evaluation: Recall")
        print("evaluating car, bike, person, bus:")

        # +++ Known - only +++
        print("evaluating car, pedestrian, truck and tram:")
        exclude_classes = ("person_sitting", "cyclist", "van", "misc")
        gt, n_gt_boxes = load_gt(exclude_classes, prefix_dir_name=label_prefix)
        evaluate_all_folders_kitti(gt, "car, ped., truck and tram (" + str(n_gt_boxes) + " bounding boxes)",
                                   output_dir=output_dir,
                                   user_specified_result_dir=FLAGS.evaluate_dir,
                                   ignored_sequences=sequences_to_ignore)

        # +++ Car - only +++
        print("evaluating car:")
        exclude_classes = ("pedestrian", "person_sitting", "cyclist", "van", "truck", "tram", "misc")
        gt, n_gt_boxes = load_gt(exclude_classes, prefix_dir_name=label_prefix)
        evaluate_all_folders_kitti(gt, "car (" + str(n_gt_boxes) + " bounding boxes)",
                                   output_dir=output_dir,
                                   user_specified_result_dir=FLAGS.evaluate_dir,
                                   ignored_sequences=sequences_to_ignore)

        # # +++ Pedestrian - only +++
        print("evaluating pedestrian:")
        exclude_classes = ("car", "person_sitting", "cyclist", "van", "truck", "tram", "misc")
        gt, n_gt_boxes = load_gt(exclude_classes, prefix_dir_name=label_prefix)
        evaluate_all_folders_kitti(gt, "pedestrian (" + str(n_gt_boxes) + " bounding boxes)",
                                   output_dir=output_dir,
                                   user_specified_result_dir=FLAGS.evaluate_dir,
                                   ignored_sequences=sequences_to_ignore)

        # +++ Misc - only +++
        print("evaluating misc:")
        exclude_classes = ("car", "pedestrian", "person_sitting", "cyclist", "van", "truck", "tram")
        gt, n_gt_boxes = load_gt(exclude_classes, prefix_dir_name=label_prefix)
        evaluate_all_folders_kitti(gt, "misc (" + str(n_gt_boxes) + " bounding boxes)",
                                   output_dir=output_dir,
                                   user_specified_result_dir=FLAGS.evaluate_dir,
                                   ignored_sequences=sequences_to_ignore)

        # # +++ "other" categories +++
        print("evaluating cyclist+van+truck+tram+misc:")
        exclude_classes = ("cyclist", "van", "truck", "tram", "misc")
        gt, n_gt_boxes = load_gt(exclude_classes, prefix_dir_name=label_prefix)
        evaluate_all_folders_kitti(gt, "cyclist+van+truck+tram+misc (" + str(n_gt_boxes) + " bounding boxes)",
                                   output_dir=output_dir,
                                   user_specified_result_dir=FLAGS.evaluate_dir,
                                   ignored_sequences=sequences_to_ignore)

        # +++ All categories +++
        print("evaluating all:")
        exclude_classes = ()
        gt, n_gt_boxes = load_gt(exclude_classes, prefix_dir_name=label_prefix)
        evaluate_all_folders_kitti(gt, "all (" + str(n_gt_boxes) + " bounding boxes)",
                                   output_dir=output_dir,
                                   user_specified_result_dir=FLAGS.evaluate_dir,
                                   sequences_to_ignore=sequences_to_ignore)
        #plt.show()


def main():

    # Matplotlib params
    matplotlib.rcParams.update({'font.size':12})
    matplotlib.rcParams.update({'font.family':'sans-serif'})
    matplotlib.rcParams['text.usetex'] = False

    # Prep output dir (if specified)
    output_dir = None
    if FLAGS.plot_output_dir is not None:
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d__%H_%M_%S')

        if FLAGS.do_not_timestamp:
            timestamp = ""

        output_dir = os.path.join(FLAGS.plot_output_dir, timestamp)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    if FLAGS.dataset == "oxford_snippets":
        eval_recall_oxford(output_dir=output_dir)
    elif FLAGS.dataset == "kitti":
        eval_recall_kitti(output_dir=output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Args
    parser.add_argument('--evaluate_recall', action='store_true', help='Run recall evaluation?')
    parser.add_argument('--evaluate_embeddings', action='store_true', help='Run embeddings evaluation?')
    parser.add_argument('--plot_output_dir', type=str, help='Plots output dir.')
    parser.add_argument('--evaluate_dir', type=str, help='Dir, containing result files that you want to evaluate')
    parser.add_argument('--always_eval_baselines', action='store_true', help='Always add baselines to the eval?')
    parser.add_argument('--labels', type=str, default = 'single_image_annotations_without_parts',
                        help='Specify dir containing the labels')
    parser.add_argument('--dataset', type=str, default = 'oxford_snippets',
                        help='Which dataset? {oxford_snippets, kitti}')
    parser.add_argument('--do_not_timestamp', action='store_true', help='Dont timestamp out dirs')
    parser.add_argument('--eval_scoring_fnc', action='store_true', help='Only eval scoring fncs')

    FLAGS = parser.parse_args()
    main()
