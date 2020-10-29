import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import torch_dct as dct

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PRN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        layers = []
        for i in range(5):
            layers.append(BasicBlock(64, 64))
        self.resnet_block = nn.Sequential(*layers)
        self.conv2 = nn.Conv2d(64, 16, kernel_size=1,
                               bias=False)
        self.conv3 = nn.Conv2d(16, 3, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.resnet_block(out)
        out = self.conv2(out)
        out = self.conv3(out)

        return out


class PRN_detector(nn.Module):
    def __init__(self, image_size):
        super().__init__()
        self.prn = PRN()
        self.fc = nn.Linear(image_size**2, 1)

    def forward(self, x):
        rect_out = self.prn(x)
        diff = x - rect_out
        diff_gs = 0.212 * diff[:, 0] + 0.715 * diff[:, 1] + 0.072 * diff[:, 2]
        diff_gs_dct = torch.log(torch.abs(dct.dct_2d(diff_gs)) + 0.0000005)
        dct_features = torch.flatten(diff_gs_dct, start_dim=1)
        out = self.fc(dct_features)
        return out
