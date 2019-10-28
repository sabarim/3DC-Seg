#Adapted converter for caffe2 weights based on the work in https://github.com/moabitcoin/ig65m-pytorch

#!/usr/bin/env python3

import pickle
import argparse
from pathlib import Path

import torch
import torch.nn as nn

from torchvision.models.video.resnet import VideoResNet, R2Plus1dStem, Conv2Plus1D, BasicBlock
from Resnet3d import resnet152_csn_ip, resnet152_csn_ir, Bottleneck

def csn_ip(pretrained=False, progress=False, **kwargs):
    model = resnet152_csn_ip(sample_size=224, sample_duration=32)

    num_classes = 400
    model.fc = nn.Linear(model.fc.in_features, out_features=num_classes)

    # We need exact Caffe2 momentum for BatchNorm scaling
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eps = 1e-3
            m.momentum = 0.9

    return model

def csn_ir(pretrained=False, progress=False, **kwargs):
    model = resnet152_csn_ir(sample_size=224, sample_duration=32)

    num_classes = 400
    model.fc = nn.Linear(model.fc.in_features, out_features=num_classes)

    # We need exact Caffe2 momentum for BatchNorm scaling
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eps = 1e-3
            m.momentum = 0.9

    return model

def blobs_from_pkl(path, num_classes=400):
    with path.open(mode="rb") as f:
        pkl = pickle.load(f, encoding="latin1")
        blobs = pkl["blobs"]

        assert "last_out_L" + str(num_classes) + "_w" in blobs, \
            "Number of --classes argument doesnt matche the last linear layer in pkl"
        assert "last_out_L" + str(num_classes) + "_b" in blobs, \
            "Number of --classes argument doesnt matche the last linear layer in pkl"

        return blobs


def copy_tensor(data, blobs, name):
    tensor = torch.from_numpy(blobs[name])

    del blobs[name]  # enforce: use at most once

    assert data.size() == tensor.size(), f"Torch tensor has size {data.size()}, while Caffe2 tensor has size {tensor.size()}"
    assert data.dtype == tensor.dtype

    data.copy_(tensor)


def copy_conv(module, blobs, prefix):
    assert isinstance(module, nn.Conv3d)
    assert module.bias is None
    copy_tensor(module.weight.data, blobs, prefix + "_w")


def copy_bn(module, blobs, prefix):
    assert isinstance(module, nn.BatchNorm3d)
    copy_tensor(module.weight.data, blobs, prefix + "_s")
    copy_tensor(module.running_mean.data, blobs, prefix + "_rm")
    copy_tensor(module.running_var.data, blobs, prefix + "_riv")
    copy_tensor(module.bias.data, blobs, prefix + "_b")


def copy_fc(module, blobs):
    assert isinstance(module, nn.Linear)
    n = module.out_features
    copy_tensor(module.bias.data, blobs, "last_out_L" + str(n) + "_b")
    copy_tensor(module.weight.data, blobs, "last_out_L" + str(n) + "_w")


# https://github.com/facebookresearch/VMZ/blob/6c925c47b7d6545b64094a083f111258b37cbeca/lib/models/r3d_model.py#L233-L275
def copy_stem(module, blobs):
    assert isinstance(module, nn.Sequential)
    assert len(module) == 4
    copy_conv(module[0], blobs, "conv1_middle")
    copy_bn(module[1], blobs, "conv1_middle_spatbn_relu")
    assert isinstance(module[2], nn.ReLU)
    copy_conv(module[3], blobs, "conv1")


# https://github.com/pytorch/vision/blob/v0.4.0/torchvision/models/video/resnet.py#L82-L114
def copy_conv2plus1d(module, blobs, i, j):
    assert isinstance(module, Conv2Plus1D)
    assert len(module) == 4
    copy_conv(module[0], blobs, "comp_" + str(i) + "_conv_" + str(j) + "_middle")
    copy_bn(module[1], blobs, "comp_" + str(i) + "_spatbn_" + str(j) + "_middle")
    assert isinstance(module[2], nn.ReLU)
    copy_conv(module[3], blobs, "comp_" + str(i) + "_conv_" + str(j))


# https://github.com/pytorch/vision/blob/v0.4.0/torchvision/models/video/resnet.py#L82-L114
def copy_basicblock(module, blobs, i):
    assert isinstance(module, BasicBlock)

    assert len(module.conv1) == 3
    assert isinstance(module.conv1[0], Conv2Plus1D)
    copy_conv2plus1d(module.conv1[0], blobs, i, 1)
    assert isinstance(module.conv1[1], nn.BatchNorm3d)
    copy_bn(module.conv1[1], blobs, "comp_" + str(i) + "_spatbn_" + str(1))
    assert isinstance(module.conv1[2], nn.ReLU)

    assert len(module.conv2) == 2
    assert isinstance(module.conv2[0], Conv2Plus1D)
    copy_conv2plus1d(module.conv2[0], blobs, i, 2)
    assert isinstance(module.conv2[1], nn.BatchNorm3d)
    copy_bn(module.conv2[1], blobs, "comp_" + str(i) + "_spatbn_" + str(2))

    if module.downsample is not None:
        assert i in [3, 7, 13]
        assert len(module.downsample) == 2
        assert isinstance(module.downsample[0], nn.Conv3d)
        assert isinstance(module.downsample[1], nn.BatchNorm3d)
        copy_conv(module.downsample[0], blobs, "shortcut_projection_" + str(i))
        copy_bn(module.downsample[1], blobs, "shortcut_projection_" + str(i) + "_spatbn")

def copy_bottleneck(module, blobs, i):
    assert isinstance(module, Bottleneck)

    copy_conv(module.conv1, blobs, "comp_" + str(i) + "_conv_" + str(1))
    copy_bn(module.bn1, blobs, "comp_" + str(i) + "_spatbn_" + str(1))

    #Adjust for a potential bug in naming of layers in Facebook net: second ID counts up
    #twice for a depthwise convolutional layer.
    copy_conv(module.conv2, blobs, "comp_" + str(i) + "_conv_" + str(3))
    copy_bn(module.bn2, blobs, "comp_" + str(i) + "_spatbn_" + str(3))

    copy_conv(module.conv3, blobs, "comp_" + str(i) + "_conv_" + str(4))
    copy_bn(module.bn3, blobs, "comp_" + str(i) + "_spatbn_" + str(4))

    if module.downsample is not None:
        assert i in [0, 3, 11, 47], str(i)
        assert len(module.downsample) == 2
        copy_conv(module.downsample[0], blobs, "shortcut_projection_" + str(i))
        copy_bn(module.downsample[1], blobs, "shortcut_projection_" + str(i) + "_spatbn")


def copy_bottleneck_csn_ip(module, blobs, i):
    assert isinstance(module, Bottleneck)

    copy_conv(module.conv1, blobs, "comp_" + str(i) + "_conv_" + str(1))
    copy_bn(module.bn1, blobs, "comp_" + str(i) + "_spatbn_" + str(1))

    copy_conv(module.conv2, blobs, "comp_" + str(i) + "_conv_" + str(2) + '_middle')
    copy_bn(module.bn2, blobs, "comp_" + str(i) + "_spatbn_" + str(2)+ '_middle')

    copy_conv(module.conv3, blobs, "comp_" + str(i) + "_conv_" + str(2))
    copy_bn(module.bn3, blobs, "comp_" + str(i) + "_spatbn_" + str(2))

    copy_conv(module.conv4, blobs, "comp_" + str(i) + "_conv_" + str(3))
    copy_bn(module.bn4, blobs, "comp_" + str(i) + "_spatbn_" + str(3))

    if module.downsample is not None:
        assert i in [0, 3, 11, 47], str(i)
        assert len(module.downsample) == 2
        copy_conv(module.downsample[0], blobs, "shortcut_projection_" + str(i))
        copy_bn(module.downsample[1], blobs, "shortcut_projection_" + str(i) + "_spatbn")


def init_canary(model):
    nan = float("nan")

    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            assert m.bias is None
            nn.init.constant_(m.weight, nan)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, nan)
            nn.init.constant_(m.running_mean, nan)
            nn.init.constant_(m.running_var, nan)
            nn.init.constant_(m.bias, nan)
        elif isinstance(m, nn.Linear):
            nn.init.constant_(m.weight, nan)
            nn.init.constant_(m.bias, nan)


def check_canary(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            assert m.bias is None
            assert not torch.isnan(m.weight).any()
        elif isinstance(m, nn.BatchNorm3d):
            assert not torch.isnan(m.weight).any()
            assert not torch.isnan(m.running_mean).any()
            assert not torch.isnan(m.running_var).any()
            assert not torch.isnan(m.bias).any()
        elif isinstance(m, nn.Linear):
            assert not torch.isnan(m.weight).any()
            assert not torch.isnan(m.bias).any()


def main(args):
    blobs = blobs_from_pkl(args.pkl)

    if(args.model == "csn_ip"):
      model = csn_ip()
    elif(args.model == "csn_ir"):
      model = csn_ir()
    else:
      raise ValueError(args.model + " is unknown")

    init_canary(model)

    copy_conv(model.conv1, blobs, "conv1")
    copy_bn(model.bn1, blobs, "conv1_spatbn_relu")

    layers = [model.layer1, model.layer2, model.layer3, model.layer4]
    blocks = [0, 3, 11, 47]

    for layer, i in zip(layers, blocks):
      assert {0: 3, 3: 8, 11: 36, 47: 3}[i] == len(layer)

      j=i
      for bottleneck in layer:
        if(args.model == 'csn_ip'):
          copy_bottleneck_csn_ip(bottleneck, blobs, j)
        else:
          copy_bottleneck(bottleneck, blobs, j)
        j += 1

    copy_fc(model.fc, blobs)

    assert not blobs
    check_canary(model)

    # Export to pytorch .pth and self-contained onnx .pb files

    torch.save(model.state_dict(), args.out.with_suffix(".pth"))
    #torch.onnx.export(model, batch, args.out.with_suffix(".pb"))

    # Check pth roundtrip into fresh model

    if(args.model == "csn_ip"):
      model = csn_ip()
    elif(args.model == "csn_ir"):
      model = csn_ir()
    model.load_state_dict(torch.load(args.out.with_suffix(".pth")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg("pkl", type=Path, help=".pkl file to read the weights from")
    arg("out", type=Path, help="prefix to save converted layer weights to")
    arg("model", choices=("csn_ip", "csn_ir"), help="model type the weights belong to")

    main(parser.parse_args())
