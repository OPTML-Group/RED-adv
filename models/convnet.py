import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, num_channels=3, num_classes=10, num_outputs=3):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # self.bn1 = nn.BatchNorm2d(32)
        # self.bn2 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(12544, 128)
        self.fc2 = nn.Linear(128, num_classes * num_outputs)
        self.num_classes = num_classes
        self.num_outputs = num_outputs

    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        # x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        # x = F.sigmoid(x)
        output = x.view([-1, self.num_classes, self.num_outputs])
        return output
