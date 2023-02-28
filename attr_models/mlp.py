import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, num_channel=3, num_class=5, num_output=3, img_size=32):
        super(MLP, self).__init__()
        self.encoder = nn.Sequential(
            Flatten(),
            nn.Linear(num_channel * img_size * img_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_class * num_output)
        )
        self.num_class = num_class
        self.num_output = num_output

    def forward(self, x):
        x = self.encoder(x)
        return x.view([-1, self.num_class, self.num_output])


class Flatten(torch.nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)
