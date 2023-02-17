'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def conv(args, channels_in, channels_out, kernel_size=3, stride=1, padding= 0,  groups=1, dilation=1):
    kernel_size = args.kernel_size
    padding = int((args.kernel_size-1)/2),
    
    return nn.Conv2d(channels_in, channels_out,
                         kernel_size=kernel_size, stride=stride, padding=padding,
                         groups=groups, bias=False, dilation=dilation)

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, args, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        # self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1 = conv(args, in_planes, planes, kernel_size=args.kernel_size, stride=stride, padding=int((args.kernel_size-1)/2)) # if ks = 3 then padding = 1; if ks = 1 then padding = 0; if ks = 5 then padding = 2
        
        self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = conv(args, planes, planes, kernel_size=args.kernel_size, stride=1, padding=int((args.kernel_size-1)/2)) # if ks = 3 then padding = 1; if ks = 1 then padding = 0; if ks = 5 then padding = 2
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )
        
        if args.activation_function == "relu":
            self.mod = nn.ReLU(inplace=True)
        elif args.activation_function == 'tanh':
            self.mod = nn.Tanh()
        elif args.activation_function == 'elu':
            self.mod = nn.ELU(inplace=True)

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        out = self.mod(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        # out = F.relu(out)
        out = self.mod(out)
        return out


class ResNet(nn.Module):
    def __init__(self, args, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        # self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = conv(args, 3, 16, kernel_size=args.kernel_size, stride=1, padding=int((args.kernel_size-1)/2)) # if ks = 3 then padding = 1; if ks = 1 then padding = 0; if ks = 5 then padding = 2
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(args, block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(args, block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(args, block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        if args.activation_function == "relu":
            self.mod = nn.ReLU(inplace=True)
        elif args.activation_function == 'tanh':
            self.mod = nn.Tanh()
        elif args.activation_function == 'elu':
            self.mod = nn.ELU(inplace=True)

        self.apply(_weights_init)

    def _make_layer(self, args, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(args, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        out = self.mod(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(args):
    return ResNet(args, BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()