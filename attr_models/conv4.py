import torch.nn as nn


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class ConvNet4(nn.Module):
    def __init__(self, num_channel=3, num_class=5, num_output=3, img_size=32):
        super().__init__()
        sz = img_size // 16
        self.encoder = nn.Sequential(
            conv_block(num_channel, 64),
            conv_block(64, 64),
            conv_block(64, 64),
            conv_block(64, 64),
            Flatten(),
            nn.Linear(sz * sz * 64, 128),
            nn.Linear(128, num_class * num_output),
        )
        self.num_class = num_class
        self.num_output = num_output

    def forward(self, x):
        x = self.encoder(x)
        return x.view([-1, self.num_class, self.num_output])


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
