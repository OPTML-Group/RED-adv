import torch
from torch import nn
import torch.nn.functional as F


class AttrNet(nn.Module):
    def __init__(self, num_channel=3, num_class=5, num_output=3, img_size=32):
        super(AttrNet, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        sz = (img_size - 3) // 2
        self.fc1 = nn.Linear(sz * sz * 64, 256)
        self.fc2 = nn.Linear(256, num_class * num_output)
        self.num_class = num_class
        self.num_output = num_output

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        outputs = x.view([-1, self.num_class, self.num_output])
        return outputs


class ConvNet2(nn.Module):
    def __init__(self, num_channel=3, num_class=5, num_output=3, img_size=32):
        sz = (img_size // 2 - 2) // 2
        super(ConvNet2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channel, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            Flatten(),
            nn.Linear(sz * sz * 64, 128),
            nn.Linear(128, num_class * num_output)
        )
        self.num_class = num_class
        self.num_output = num_output

    def forward(self, x):
        x = self.encoder(x)
        outputs = x.view([-1, self.num_class, self.num_output])
        return outputs

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)
