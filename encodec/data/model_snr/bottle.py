import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class BottleNeck1d(nn.Module):
    """
    ResNet 2 conv residual block
    batchnorm + preactivation
    dropout used when net is wide
    """

    def __init__(self, in_channels, out_channels, stride, kernel_size, use_bn=True):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bn = use_bn

        self.pre_act = nn.BatchNorm1d(in_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2
        )
        self.bn = nn.BatchNorm1d(out_channels)

        if in_channels >= 128 or out_channels >= 128:
            p = 0.05
            if in_channels >= 256 or out_channels >= 256:
                p = 0.1
            if in_channels > 256 or out_channels > 256:
                p = 0.25
            self.dropout = nn.Dropout(p=p)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2
        )

    def forward(self, x):
        x = F.relu(self.pre_act(x))
        if self.stride != 1 or self.in_channels != self.out_channels:
            y = self.shortcut(x)
        else:
            y = x
        x = F.relu(self.bn(self.conv1(x))) if self.use_bn else F.relu(self.conv1(x))
        if self.in_channels > 128 or self.out_channels > 128:
            x = self.dropout(x)
        x = self.conv2(x)
        x = x + y
        return x


class BottleNeck2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size, dropout_prob=0.1, bn=True):
        super(BottleNeck2d, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.pre_act = nn.BatchNorm2d(in_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
        if type(kernel_size) in [tuple, list]:
            pad_num = tuple((np.array(kernel_size) - 1) // 2)
        else:
            pad_num = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=pad_num)
        self.bn = nn.BatchNorm2d(out_channels) if bn else lambda x: x
        self.dropout = nn.Dropout2d(p=dropout_prob)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=pad_num)

    def forward(self, x):
        x = F.relu(self.pre_act(x))
        if self.stride != 1 or self.in_channels != self.out_channels:
            y = self.shortcut(x)
        else:
            y = x
        x = self.dropout(F.relu(self.bn(self.conv1(x))))
        x = self.conv2(x)
        x = x + y
        return x