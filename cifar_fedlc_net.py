from fluke.nets import EncoderHeadNet

"""
This module contains the definition of several neural networks used in state-of-the-art
federated learning papers.
"""
import sys
import torch.nn as nn

sys.path.append(".")
sys.path.append("..")

class EncoderBN(nn.Module):
    def __init__(self, input_nc, img_size=32):
        super().__init__()
        self.image_size = img_size
        self.features = nn.Sequential(
            # conv1
            nn.Conv2d(input_nc, img_size, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(img_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # conv2
            nn.Conv2d(img_size, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        for layer in self.features:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        x = self.features(x)
        out = x.view(x.size(0), -1)
        return out

class ClassifierBN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.output_size = num_classes

        self.classifier = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        return self.classifier(x)

class FedLCNet(EncoderHeadNet):
    def __init__(self, channels: int = 3):
        super(FedLCNet, self).__init__(EncoderBN(channels), ClassifierBN())