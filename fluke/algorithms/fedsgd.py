"""Implementation of the [FedSGD17]_ algorithm.

References:
    .. [FedSGD17] H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera
       y Arcas. Communication-efficient learning of deep networks from decentralized data.
       In AISTATS (2017). URL: https://arxiv.org/abs/1602.05629
"""
import torch
import sys
sys.path.append(".")
sys.path.append("..")

from ..algorithms import CentralizedFL  # NOQA
from ..data import FastDataLoader  # NOQA
from ..utils import OptimizerConfigurator  # NOQA
from ..client import Client  # NOQA


class ClientSGD(Client):
    def __init__(self,
                 index: int,
                 train_set: FastDataLoader,
                 test_set: FastDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: torch.nn.Module,
                 local_epochs: int = 3,
                 **kwargs):
        super().__init__(index=index, train_set=train_set, test_set=test_set,
                         optimizer_cfg=optimizer_cfg, loss_fn=loss_fn, local_epochs=local_epochs,
                         **kwargs)
        self.train_set.single_batch = True
        self.train_set.shuffle = True
        self.hyper_params.local_epochs = 1


class FedSGD(CentralizedFL):

    def get_client_class(self) -> Client:
        return ClientSGD
