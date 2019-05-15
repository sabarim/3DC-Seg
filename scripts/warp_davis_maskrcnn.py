import glob
import os
import sys
from math import ceil
import pickle
import cv2
import numpy as np
import torch
from PIL import Image
from scipy.misc import imread, imsave
from scipy.ndimage import imread
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm
from shutil import copy

from lib.flowlib import read_flow
from structures.bounding_box import BoxList
from util import save_mask, select_top_predictions


CONF_THRESH=0.3
maskrcnn_data_dir = "/globalwork/mahadevan/mywork/data/training/pytorch/forwarded/maskrcnn/"
davis_data_dir = '/globalwork/data/DAVIS-Unsupervised/DAVIS/'
flow_dir = "/globalwork/data/DAVIS-Unsupervised/DAVIS/flo/"
out_dir = "../results/maskrcnn_warped/"


def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow

    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()
    # TODO: check if multiplying with -1 is the right thing to do for forward warp
    vgrid = Variable(grid) + Variable(flo*-1)

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid.cuda())
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    mask = nn.functional.grid_sample(mask, vgrid)
    # mask = nn.functional.interpolate(mask, mode='bilinear', size = mask.shape[2:])

    # if W==128:
    # np.save('mask.npy', mask.cpu().data.numpy())
    # np.save('warp.npy', output.cpu().data.numpy())

    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    return output * mask


def save_mask_warp(proposals, out_folder, flow_fn):
    # rescale the image size to be multiples of 64
    divisor = 64.
    flo = torch.from_numpy(read_flow(flow_fn))[None]
    top_predictions = select_top_predictions(proposals, CONF_THRESH)
    result_predictions = {}
    for i in range(len(top_predictions.get_field('scores'))):
        result_predictions[i] = {'mask': warp(top_predictions.get_field('mask')[i:i+1].float().cuda(), flo.permute(0,-1,1,2).cuda()),
                                 'score': top_predictions.get_field('scores')[i]}

    video = flow_fn.split("/")[-2]
    out_file = os.path.join(out_folder, os.path.basename(flow_fn).replace(".flo", ".pickle"))
    os.makedirs(os.path.dirname(out_file)) if not os.path.exists(os.path.dirname(out_file)) else None
    pickle.dump(result_predictions, open(out_file, 'wb'))


def main():
    seqs = davis_data_dir + "ImageSets/2017/val.txt"
    with open(os.path.join(seqs), "r") as lines:
        for line in lines:
            print(line)
            line = line.rstrip()
            out_folder = out_dir + line
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            pickle_files = glob.glob(os.path.join(maskrcnn_data_dir, line, "*.pickle"))

            for i in range(len(pickle_files) - 1):
                flow_fn = os.path.join(flow_dir, line, '{:05d}'.format(i+1) + ".flo")
                proposals = pickle.load(open(os.path.join(maskrcnn_data_dir,
                                                          line, '{:05d}'.format(i) + ".pickle"), 'rb'))
                save_mask_warp(proposals, out_folder, flow_fn)

if __name__ == '__main__':
    main()



