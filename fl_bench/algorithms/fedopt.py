from enum import Enum
from copy import deepcopy
from collections import OrderedDict
from typing import Callable, Iterable, Union, Optional, Any

import torch
from torch.nn import Module

import sys
from fl_bench.data import FastTensorDataLoader; sys.path.append(".")
from fl_bench.server import Server
from fl_bench.client import Client
from fl_bench.utils import DDict, OptimizerConfigurator
from fl_bench.algorithms import CentralizedFL

class FedOptMode(Enum):
    FedAdam = "adam"
    FedYogi = "yogi"
    FedAdagrad = "adagrad"


class FedOptServer(Server):
    def __init__(self,
                 model: Module,
                 test_data: FastTensorDataLoader,
                 clients: Iterable[Client], 
                 mode: str="fedadam",
                 lr: float=0.001,
                 beta1: float=0.9,
                 beta2: float=0.999,
                 tau: float=0.0001,
                 weighted: bool=True):
        super().__init__(model, test_data, clients, weighted)
        # assert mode in FedOptMode, "mode must be one of FedOptMode"
        assert 0 <= beta1 < 1, "beta1 must be in [0, 1)"
        assert 0 <= beta2 < 1, "beta2 must be in [0, 1)"
        self.mode = mode if isinstance(mode, FedOptMode) else FedOptMode(mode)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.tau = tau
        self._init_moments()

    def _init_moments(self):
        self.m = OrderedDict()
        self.v = OrderedDict()
        for key in self.model.state_dict().keys():
            if not "num_batches_tracked" in key:
                self.m[key] = torch.zeros_like(self.model.state_dict()[key])
                # This guarantees that the second moment is >= 0 and <= tau^2
                self.v[key] = torch.rand_like(self.model.state_dict()[key]) * self.tau ** 2
    
    def aggregate(self, eligible: Iterable[Client]) -> None:
        avg_model_sd = OrderedDict()
        clients_sd = self._get_client_models(eligible)
        with torch.no_grad():
            for key in self.model.state_dict().keys():
                if "num_batches_tracked" in key:
                    avg_model_sd[key] = deepcopy(clients_sd[0][key])
                    continue

                den, diff = 0, 0
                for i, client_sd in enumerate(clients_sd):
                    weight = 1 if not self.weighted else eligible[i].n_examples
                    diff += weight * (client_sd[key] - self.model.state_dict()[key])
                    den += weight
                diff /= den
                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * diff

                diff_2 = diff ** 2
                if self.mode == FedOptMode.FedAdam:
                    self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * diff_2
                elif self.mode == FedOptMode.FedYogi:
                    self.v[key] -= (1 - self.beta2) * diff_2 * torch.sign(self.v[key] - diff_2)
                elif self.mode == FedOptMode.FedAdagrad:
                    self.v[key] += diff_2
                    
                update = self.m[key] + self.lr * self.m[key] / (torch.sqrt(self.v[key]) + self.tau)
                avg_model_sd[key] = self.model.state_dict()[key] + update
            
            self.model.load_state_dict(avg_model_sd)
    
    def __str__(self) -> str:
        to_str = super().__str__()
        return f"{to_str[:-1]},mode={self.mode},lr={self.lr},beta1={self.beta1},beta2={self.beta2},tau={self.tau})"


class FedOpt(CentralizedFL):
    
    def init_server(self, model: Any, data: FastTensorDataLoader, config: DDict):
        self.server = FedOptServer(model, data, self.clients, **config)

    

