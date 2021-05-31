from typing_extensions import runtime
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
import torchvision.transforms as transforms
import cv2

class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(10):
            layers.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(64))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, imgR):
        result = self.dncnn(imgR)+imgR
        return result