from .channel_estimation_model import  SimSiam_Channel

import torch

from torch.nn.modules import Module

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
from torch import nn

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from tqdm import tqdm

from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import random

import copy
from torch import nn

torch.manual_seed(1)
torch.cuda.manual_seed(1)
random.seed(0)
np.random.seed(0)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return out


class DnCNN_MultiBlock(nn.Module):
    def __init__(self, block, depth, image_channels, filters=64,  use_bnorm=True):
        super(DnCNN_MultiBlock, self).__init__()


        self.layer1 = self._make_layer(block, depth, filters)
        self.noise_layer1 = nn.Conv2d(filters, 2, kernel_size=3, stride=1, padding=1, bias=False)

        self.layer2 = self._make_layer(block, depth, filters)
        self.noise_layer2 = nn.Conv2d(filters, 2, kernel_size=3, stride=1, padding=1, bias=False)


    def _make_layer(self, block, depth, planes):
        layers = []
        for i in range(depth-1):

            if i == 0 :
                in_planes = 2
            else:
                in_planes = planes

            layers.append(block(in_planes, planes))
        return nn.Sequential(*layers)



    def forward(self, x, ls):

        out = self.layer1(ls)
        out1 = self.noise_layer1(out)

        out1[:,0,:,:] = 0.5*ls[:,1,:,:].clone() + 0.5*F.tanh(out1[:,0,:,:]).clone()
        out1[:,1,:,:] = 0.5*ls[:,0,:,:].clone() + 0.5*F.tanh(out1[:,1,:,:]).clone()

        out = self.layer2(out1)
        out = self.noise_layer2(out)
        out = ls - F.tanh(out)

        return out


def DnCNN_MultiBlock_v1(name=None):
    return DnCNN_MultiBlock(BasicBlock, depth=5, image_channels=2, filters=64, use_bnorm=True)

class Identity(Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input



def get_model(model_cfg, train_cnn=0):

    if model_cfg.name == 'IRS_CE_model':

        if train_cnn:
            model = DnCNN_MultiBlock_v1()
        else:
            model = SimSiam_Channel(DnCNN_MultiBlock_v1())

    return model
