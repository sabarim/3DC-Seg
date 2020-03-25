import argparse
import glob
import os
import numpy as np

from PIL import Image
from sklearn.metrics import precision_recall_curve


def main(args):
    results_path = args.results_path
    gt_path = args.gt_path

    gt_seqs = glob.glob(gt_path + "/*")
    result_seqs = glob.glob(results_path + "/*")
    if len(result_seqs) < gt_seqs:
        print("WARN: The results do not have all the ground truth sequences.")

    for seq in gt_seqs:
        seq_name = seq.split("/")[-2]
        result_seq_path = os.path.join(results_path, seq_name)
        preds = []
        gts = []
        if not os.path.exists(result_seq_path):
            print("Sequence {} does not exist in the results path.".format(seq_name))
        for f in glob.glob(result_seq_path + "/*.png"):
            f_name = f.split("/")[-1]
            preds+=[np.array(Image.open(f).convert("P"))]
            gt = np.array(Image.open(os.path.join(seq, f_name)))
            gts+=[(gt!=0).astype(np.uint8)]

        pred = np.stack(preds).flatten()
        gt = np.stack(gts).flatten()
        precision, recall, _= precision_recall_curve(gt, pred)
        Fmax = 2 * (precision * recall) / (precision + recall)
        print('MAE', np.mean(abs(pred - gt)))
        print('F_max', Fmax.max())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval_fbms')
    parser.add_argument('--results_path', dest='results_path',
                        help='results path',
                        type=str, required=True)
    parser.add_argument('--gt_path', dest='gt_path',
                        help='ground truth path',
                        type=str, required=True)
    main(parser.parse_args())