import torch
import torch.nn.functional as F
from torch import nn


class LeNet(nn.Module):
    def __init__(self, num_channel=3, num_class=5, num_output=3, img_size=32):
        super(LeNet, self).__init__()
        pd = max(0, (32 - img_size) // 2)
        self.conv1 = nn.Conv2d(num_channel, 6, kernel_size=5, padding=pd)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        sz = ((img_size + pd * 2 - 3) // 2 - 3) // 2
        self.fc1 = nn.Linear(16 * sz * sz, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_class * num_output)
        self.num_class = num_class
        self.num_output = num_output

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view([-1, self.num_class, self.num_output])
