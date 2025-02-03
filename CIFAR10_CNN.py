import torch
import torch.nn as nn
import torch.nn.functional as F
from fluke.nets import EncoderHeadNet

__all__ = ["CIFAR_CNN"]

class CIFAR_CNN_E(nn.Module):
    """Encoder for the :class:`CIFAR_CNN` network.

    See Also:
        - :class:`CIFAR_CNN`
        - :class:`CIFAR_CNN_D`
    """

    def __init__(self):
        super(CIFAR_CNN_E, self).__init__()
        self.output_size = 1600
        self.conv1 = nn.Conv2d(3, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        return x.view(-1, 1600)


class CIFAR_CNN_D(nn.Module):
    """Head for the :class:`CIFAR_CNN` network.

    See Also:
        - :class:`CIFAR_CNN`
        - :class:`CIFAR_CNN_E`
    """

    def __init__(self):
        super(CIFAR_CNN_D, self).__init__()
        self.output_size = 100
        self.fc1 = nn.Linear(1600, 512)
        self.fc2 = nn.Linear(512, 100)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


# FedAvg: https://arxiv.org/pdf/1602.05629.pdf
# SuPerFed - https://arxiv.org/pdf/2109.07628v3.pdf
# works with 1 channel input - MNIST4D
class CIFAR_CNN(EncoderHeadNet):
    """Convolutional Neural Network for MNIST. This is a simple CNN for MNIST classification
    first introduced in the [FedAvg]_ paper, where the architecture consists of two convolutional
    layers with 32 and 64 filters, respectively, followed by two fully connected layers with 512
    and 10 neurons, respectively.

    Very same architecture is also used in the [SuPerFed]_ paper.
    """

    def __init__(self):
        super(CIFAR_CNN, self).__init__(CIFAR_CNN_E(), CIFAR_CNN_D())