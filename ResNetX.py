import inspect
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from sklearn.decomposition import PCA

from fluke.nets import EncoderHeadNet

__all__ = ["ResNet8"]

class Reshape(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, xs):
        return xs.reshape((xs.shape[0], -1))

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes,
            kernel_size=3, stride=stride,
            padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes, planes,
            kernel_size=3, stride=1,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ 6n + 2: 8, 14, 20, 26, 32, 38, 44, 50, 56
    """

    def __init__(self, n_layer=8, n_classes=10):
        super().__init__()
        self.n_layer = n_layer
        self.n_classes = n_classes

        conv1 = nn.Conv2d(
            3, 16, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        bn1 = nn.BatchNorm2d(16)

        assert ((n_layer - 2) % 6 == 0), "SmallResNet depth is 6n+2"
        n = int((n_layer - 2) / 6)

        self.cfg = (BasicBlock, (n, n, n))
        self.in_planes = 16

        layer1 = self._make_layer(
            block=self.cfg[0], planes=16, stride=1, num_blocks=self.cfg[1][0],
        )
        layer2 = self._make_layer(
            block=self.cfg[0], planes=32, stride=2, num_blocks=self.cfg[1][1],
        )
        layer3 = self._make_layer(
            block=self.cfg[0], planes=64, stride=2, num_blocks=self.cfg[1][2],
        )

        avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.encoder = nn.Sequential(
            conv1,
            bn1,
            nn.ReLU(True),
            layer1,
            layer2,
            layer3,
            avgpool,
            Reshape(),
        )

        self.h_size = 64 * self.cfg[0].expansion
        
        self.classifier = nn.Linear(
            64 * self.cfg[0].expansion, n_classes
        )

    def _make_layer(self, block, planes, stride, num_blocks):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = block.expansion * planes
        return nn.Sequential(*layers)

    def forward(self, xs):
        hs = self.encoder(xs)
        logits = self.classifier(hs)
        return hs, logits