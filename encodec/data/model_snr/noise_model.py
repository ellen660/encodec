import torch
import torch.nn as nn
import torch.nn.functional as F

from bottle import BottleNeck1d, BottleNeck2d

class NoiseDetectionModel(nn.Module):
    SignalDuration = 600
    SignalFS = 5
    SignalClipLimit = 5

    def __init__(self):
        super().__init__()
        self.capacity = 32

        self.conv_sig = nn.Conv1d(1, 8, kernel_size=13, stride=2, padding=6)

        self.block1_sig = BottleNeck1d(8, 16, 5, 9)
        self.block2_sig = BottleNeck1d(16, self.capacity, 3, 9)
        self.block3_sig = BottleNeck1d(self.capacity, self.capacity, 3, 7)
        self.block4_sig = BottleNeck1d(self.capacity, self.capacity, 3, 7)

        self.fc = nn.Linear(self.capacity, self.capacity)
        self.bn = nn.BatchNorm1d(self.capacity)
        self.dropout = nn.Dropout(0.5)

        self.fc_final = nn.Linear(self.capacity, 1)

    def forward(self, x: torch.Tensor):
        x = self.conv_sig(x)
        x = self.block1_sig(x)
        x = self.block2_sig(x)
        x = self.block3_sig(x)
        x = self.block4_sig(x)
        x = F.max_pool1d(x, kernel_size=3)
        x = x.view(-1, self.capacity)

        x = self.dropout(F.relu(self.bn(self.fc(x))))

        y = torch.sigmoid(self.fc_final(x))
        return y, x

    @classmethod
    def load_model(cls):
        pass


class NoiseDetectionModelWithSpec(nn.Module):
    SignalDuration = 600
    SignalFS = 5
    SignalClipLimit = 5

    def __init__(self):
        super().__init__()
        self.capacity = 32

        self.conv_sig = nn.Conv1d(1, 8, kernel_size=13, stride=2, padding=6)
        self.block1_sig = BottleNeck1d(8, 16, 5, 9)
        self.block2_sig = BottleNeck1d(16, self.capacity, 3, 9)
        self.block3_sig = BottleNeck1d(self.capacity, self.capacity, 3, 7)
        self.block4_sig = BottleNeck1d(self.capacity, self.capacity, 3, 7)

        self.conv_spec = nn.Conv2d(1, 8, kernel_size=[15, 15], stride=[3, 3])
        self.block1_spec = BottleNeck2d(8, 16, 3, 5)
        self.block2_spec = BottleNeck2d(16, self.capacity, 3, 5)

        self.fc = nn.Linear(2 * self.capacity, 2 * self.capacity)
        self.bn = nn.BatchNorm1d(2 * self.capacity)
        self.dropout = nn.Dropout(0.5)

        self.fc_final = nn.Linear(2 * self.capacity, 1)

    def forward(self, x: torch.Tensor, z: torch.Tensor):
        z = self.conv_spec(z)
        z = self.block1_spec(z)
        z = self.block2_spec(z)
        z = F.max_pool2d(z, kernel_size=[3, 3])
        z = z.view(-1, self.capacity)

        x = self.conv_sig(x)
        x = self.block1_sig(x)
        x = self.block2_sig(x)
        x = self.block3_sig(x)
        x = self.block4_sig(x)
        x = F.max_pool1d(x, kernel_size=3)
        x = x.view(-1, self.capacity)

        x = torch.cat((x, z), dim=1)
        x = self.dropout(F.relu(self.bn(self.fc(x))))

        y = torch.sigmoid(self.fc_final(x))
        return y, x

    @classmethod
    def load_model(cls):
        pass