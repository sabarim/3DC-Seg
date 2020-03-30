import argparse
import glob
import os
import pickle
import torch
import tqdm
import numpy as np
from torch.nn import functional as F
from PIL import Image
from sklearn.metrics import precision_recall_curve


def main(args):
    results_path = args.results_path
    gt_path = args.gt_path
    F = []
    maes = []

    if args.dataset == "FBMS":
        gt_seqs = glob.glob(gt_path + "/*")
    elif args.dataset == "DAVIS":
        assert args.imageset is not None
        imageset = open(args.imageset)
        gt_seqs = [os.path.join(gt_path, s) for s in imageset.readlines()]
    result_seqs = glob.glob(results_path + "/*")
    if len(result_seqs) < len(gt_seqs):
        print("WARN: The results do not have all the ground truth sequences.")
    preds = []
    gts = []
    logits = []
    for i in tqdm.tqdm(range(len(gt_seqs))):
        seq = gt_seqs[i]
        seq_name = seq.split("/")[-1]
        result_seq_path = os.path.join(results_path, seq_name)
        gt_files = glob.glob(seq + "/*.png")
        f_num_length = len(os.path.basename(gt_files[0]).split("_")[-1].split(".")[0])
        if not os.path.exists(result_seq_path):
            print("Sequence {} does not exist in the results path.".format(seq_name))
            continue
        for f in glob.glob(os.path.join(result_seq_path, "*.png")):
            f_name = f.split("/")[-1]
            f_num = int(f_name.split(".")[0])
            gt_file = os.path.join(seq, seq_name +  ('_{:0' + str(f_num_length) + 'd}.png').format(f_num))
            if os.path.exists(gt_file):
                preds += [np.array(Image.open(f).convert("P")).flatten()]
                gt = np.array(Image.open(gt_file))
                prob = np.array(pickle.load(open(f.replace('png', 'pkl'), 'rb')))
                prob = torch.nn.functional.interpolate(torch.from_numpy(prob[None, None]), gt.shape,
                                                mode='bilinear').numpy()[0,0]
                logits += [prob.flatten()]
                gts+=[(gt!=0).astype(np.uint8).flatten()]
        print('Sequence {}'.format(seq_name))

    pred = np.hstack(preds).flatten()
    gt = np.hstack(gts).flatten()
    logits = np.hstack(logits).flatten()
    precision, recall, _= precision_recall_curve(gt, pred)
    Fmax = 2 * (precision * recall) / (precision + recall)
    F+=[Fmax.max()]
    maes += [np.mean(abs(logits - gt))]
        # print('Sequence {}: F_max {}  MAE {}'.format(seq_name, Fmax.max(), np.mean(abs(pred - gt))))

    print("Finished eval: F {}  MAE {}".format(np.mean(F), np.mean(maes)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval_fbms')
    parser.add_argument('--results_path', dest='results_path',
                        help='results path',
                        type=str, required=True)
    parser.add_argument('--gt_path', dest='gt_path',
                        help='ground truth path',
                        type=str, required=True)
    parser.add_argument('--dataset', dest='dataset',
                        help='dataset to evaluate',
                        type=str, required=True)
    parser.add_argument('--imageset', dest='imageset',
                        help='davis imageset file',
                        type=str, required=False, default=None)
    main(parser.parse_args())