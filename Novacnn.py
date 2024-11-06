'''
For the image datasets, we use a CNN, which has two 5x5 convolution layers 
followed by 2x2 max pooling (the first with 6 channels and the second with 16 channels)
and two fully connected layers with ReLU activation (the first with 120 units and the second with 84 units).
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from fluke.nets import EncoderHeadNet

__all__ = ["NovaCNN"]

class NovaCNN_E(nn.Module):
    def __init__(self):
        super(NovaCNN_E, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, 1)  # possibilie dover cambiare il 6 in base al numero di canali(X3)
        self.conv2 = nn.Conv2d(6, 16, 5, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        
        return x.view(-1, 16 * 5 * 5)  # Updated to match the calculated output size
    
class NovaCNN_D(nn.Module):

    def __init__(self):
        super(NovaCNN_D, self).__init__()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1)

class NovaCNN(EncoderHeadNet):

    def __init__(self):
        super(NovaCNN, self).__init__(NovaCNN_E(), NovaCNN_D())