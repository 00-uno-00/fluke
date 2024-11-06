import sys
import torch
import torch
import torch.nn as nn
from torchvision.models import vgg11

sys.path.append(".")
sys.path.append("..")


__all__ = ['VGG11']

class VGG11(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(VGG11, self).__init__()
        self.vgg11 = vgg11(num_classes=num_classes, pretrained=False)

    def forward(self, x) -> torch.Tensor:
        return self.vgg11(x)