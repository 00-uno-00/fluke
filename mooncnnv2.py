"""
This module contains the definition of several neural networks used in state-of-the-art
federated learning papers.
"""
import string
import sys
from abc import abstractmethod

import torch
import torch.nn as nn
from torch.functional import F
from torchvision.models import resnet18, resnet34, resnet50, vgg11

sys.path.append(".")
sys.path.append("..")

__all__ = [
    "EncoderHeadNet",
    "MoonCNN",
    "MoonCNN_D",
    "MoonCNN_E",
]

class EncoderHeadNet(nn.Module):
    r"""Encoder (aka backbone) + Head Network [Base Class]
    This type of networks are defined as two subnetworks, where one is meant to be the
    encoder/backbone network that learns a latent representation of the input, and the head network
    that is the classifier part of the model. The forward method should work as usual (i.e.,
    :math:`g(f(\mathbf{x}))` where :math:`\mathbf{x}` is the input, :math:`f` is the encoder and
    :math:`g` is the head), but the ``forward_encoder`` and ``forward_head`` methods should be used
    to get the output of the encoder and head subnetworks, respectively.
    If this is not possible, they fallback to the forward method (default behavior).

    Attributes:
        output_size (int): Output size of the head subnetwork.

    Args:
        encoder (nn.Module): Encoder subnetwork.
        head (nn.Module): Head subnetwork.
    """

    def __init__(self, encoder: nn.Module, head: nn.Module):
        super(EncoderHeadNet, self).__init__()
        self.output_size = head.output_size
        self._encoder = encoder
        self._head = head

    @property
    def encoder(self) -> nn.Module:
        """Return the encoder subnetwork.

        Returns:
            nn.Module: Encoder subnetwork.
        """
        return self._encoder

    @property
    def head(self) -> nn.Module:
        """Return the head subnetwork.

        Returns:
            nn.Module: head subnetwork.
        """
        return self._head

    def forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder subnetwork.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor of the encoder subnetwork.
        """
        return self._encoder(x)

    def forward_head(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through the head subnetwork. ``z`` is assumed to be the output of the
        encoder subnetwork or an "equivalent" tensor.

        Args:
            z (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor of the head subnetwork.
        """
        return self._head(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._head(self._encoder(x))

class MoonCNN_E(nn.Module):
    """Encoder for the :class:`MoonCNN` network.

    See Also:
        - :class:`MoonCNN`
        - :class:`MoonCNN_D`
    """

    # Expected input size: 32x32x3
    def __init__(self):
        super(MoonCNN_E, self).__init__()
        self.output_size = 400

        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)

    def forward(self, x) -> torch.Tensor:
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 400)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class MoonCNN_D(nn.Module):
    """Head for the :class:`MoonCNN` network.

    See Also:
        - :class:`MoonCNN`
        - :class:`MoonCNN_E`
    """

    def __init__(self):
        super(MoonCNN_D, self).__init__()
        self.output_size = 10
        self.out = nn.Linear(84, self.output_size)

    def forward(self, x) -> torch.Tensor:
        x = self.out(x)
        return x
    
class MoonCNN(EncoderHeadNet):
    """Convolutional Neural Network for CIFAR-10. This is a CNN for CIFAR-10 classification first
    described in the [MOON]_ paper, where the architecture consists of two convolutional layers with
    6 and 16 filters, respectively, followed by two fully connected layers with 120 and 84 neurons,
    respectively, and a projection head with 256 neurons followed by the output layer with 10
    neurons.

    See Also:
        - :class:`MoonCNN_E`
        - :class:`MoonCNN_D`

    References:
        .. [MOON] Qinbin Li, Bingsheng He, and Dawn Song. Model-Contrastive Federated Learning.
            In CVPR (2021).
    """

    def __init__(self):
        super(MoonCNN, self).__init__(MoonCNN_E(), MoonCNN_D())