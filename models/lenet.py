from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import os

class LeNet(nn.Module):
    def __init__(self, args):
        super(LeNet, self).__init__()
        self.num_conv = args.num_conv
        self.ks = args.kernel_size
        self.num_fc = args.num_fc
        dict_act_func = {"relu": F.relu, "tanh": F.tanh, "silu": F.silu, "elu": F.elu}
        self.act_func = dict_act_func[args.act_func]
        self.conv1 = nn.Conv2d(1, 32, self.ks, 1)
        self.conv2 = nn.Conv2d(32, 32, self.ks, 1)
        self.conv3 = nn.Conv2d(32, 32, self.ks, 1)
        self.conv4 = nn.Conv2d(32, 32, self.ks, 1)
        self.conv5 = nn.Conv2d(32, 32, self.ks, 1)
        self.conv6 = nn.Conv2d(32, 32, self.ks, 1)
        self.fc1 = nn.Linear(32*(28-(self.ks-1)*self.num_conv)**2, 64)
        self.fc5 = nn.Linear(64, 10)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.conv_list = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6]
        self.fc_list = [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm2d(32)
        self.bn6 = nn.BatchNorm2d(32)
        self.shortcut = nn.Sequential()
        self.bn_list = [self.bn1, self.bn2, self.bn3, self.bn4, self.bn5, self.bn6]
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act_func(x)
        for i in range(2, self.num_conv+1):
            x = self.conv_list[i-1](x)
            x = self.bn_list[i-1](x)
            x = self.act_func(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.act_func(x)
        for i in range(2, self.num_fc):
            x = self.fc_list[i-1](x)
            x = self.act_func(x)
        x = self.fc5(x)
        return x
