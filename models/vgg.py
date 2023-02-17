import torch
import torch.nn as nn


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def conv(args, channels_in, channels_out, kernel_size=3, stride=1, padding= 0,  groups=1, dilation=1):
    kernel_size = args.kernel_size
    padding = int((args.kernel_size-1)/2),
    
    return nn.Conv2d(channels_in, channels_out,
                         kernel_size=kernel_size, stride=stride, padding=padding,
                         groups=groups, bias=False, dilation=dilation)

class VGG(nn.Module):
    def __init__(self, args, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(args, cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, args, cfg):
        layers = []
        in_channels = 3
        if args.act_func == "relu":
            self.mod = nn.ReLU(inplace=True)
        elif args.act_func == 'tanh':
            self.mod = nn.Tanh()
        elif args.act_func == 'elu':
            self.mod = nn.ELU(inplace=True)
        
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [conv(args, in_channels, x, kernel_size=args.kernel_size, padding=int((args.kernel_size-1)/2)),
                           nn.BatchNorm2d(x),
                           self.mod]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
