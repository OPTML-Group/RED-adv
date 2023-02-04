import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet9(nn.Module):
    def __init__(self, num_classes = 10, kernel_size = 3, act_func = "relu"):
        super(ResNet9, self).__init__()
        self.model = torch.nn.Sequential(
            conv_bn(3, 64, kernel_size = kernel_size, stride=1, padding=int((kernel_size-1)/2), act_func=act_func), # if ks = 3 then padding = 1; if ks = 1 then padding = 0; if ks = 5 then padding = 2
            conv_bn(64, 128, kernel_size=kernel_size, stride=2, padding=int((kernel_size+1)/2), act_func=act_func), # if ks = 3 then padding = 2; if 
            Residual(torch.nn.Sequential(conv_bn(128, 128, kernel_size = kernel_size,padding=int((kernel_size-1)/2), act_func=act_func), conv_bn(128, 128,kernel_size = kernel_size,padding=int((kernel_size-1)/2), act_func=act_func))),
            conv_bn(128, 256, kernel_size=kernel_size, stride=1, padding=int((kernel_size-1)/2), act_func=act_func),
            torch.nn.MaxPool2d(2),
            Residual(torch.nn.Sequential(conv_bn(256, 256, kernel_size = kernel_size,padding=int((kernel_size-1)/2),act_func=act_func), conv_bn(256, 256, kernel_size = kernel_size,padding=int((kernel_size-1)/2), act_func=act_func))),
            conv_bn(256, 128, kernel_size=kernel_size, stride=1, padding=max(0,int((kernel_size-3)/2)), act_func=act_func),
            torch.nn.AdaptiveMaxPool2d((1, 1)),
            Flatten(),
            torch.nn.Linear(128, num_classes, bias=False),
            Mul(0.2)
        )

    def forward(self, x):
        return self.model(x)
        
def conv_bn(channels_in, channels_out, kernel_size, stride=1, padding = 1, groups=1, act_func="relu"):
    assert act_func in ["relu", "tanh", "elu"]

    if act_func == "relu":
        mod = torch.nn.ReLU(inplace=True)
    elif act_func == 'tanh':
        mod = torch.nn.Tanh()
    elif act_func == 'elu':
        mod = torch.nn.ELU(inplace=True)
    
    return torch.nn.Sequential(
        torch.nn.Conv2d(channels_in, channels_out,
                        kernel_size=kernel_size, stride=stride, padding=padding,
                        groups=groups, bias=False),
        torch.nn.BatchNorm2d(channels_out),
        mod
    )

class Mul(torch.nn.Module):
    def __init__(self, weight):
        super(Mul, self).__init__()
        self.weight = weight
    def forward(self, x): return x * self.weight

class Flatten(torch.nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

class Residual(torch.nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module
    def forward(self, x): return x + self.module(x)