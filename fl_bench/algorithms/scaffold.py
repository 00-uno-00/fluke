from typing import Callable, Iterable
from copy import deepcopy

import torch
from torch.nn import Module
from torch.optim import Optimizer

from client import Client
from fl_bench import Message
from fl_bench.client import Client
from server import Server

import sys; sys.path.append(".")
from fl_bench.algorithms import CentralizedFL
from fl_bench.utils import OptimizerConfigurator
from fl_bench.data import DataSplitter, FastTensorDataLoader


class ScaffoldOptimizer(Optimizer):
    def __init__(self, params, lr: float=0.001, weight_decay: float=0.01):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(ScaffoldOptimizer, self).__init__(params, defaults)

    #TODO: add types
    def step(self, server_controls, client_controls, closure=None):

        loss = None
        if closure is not None:
            loss = closure

        for group in self.param_groups:
            for p, c, ci in zip(group['params'], server_controls, client_controls):
                if p.grad is None:
                    continue
                dp = p.grad.data + c.data - ci.data
                p.data = p.data - dp.data * group['lr']

        return loss


class ScaffoldClient(Client):
    def __init__(self,
                 train_set: FastTensorDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: Callable,
                 validation_set: FastTensorDataLoader=None,
                 local_epochs: int=3):
        assert optimizer_cfg.optimizer == ScaffoldOptimizer, \
            "ScaffoldClient only supports ScaffoldOptimizer"
        super().__init__(train_set, optimizer_cfg, loss_fn, validation_set, local_epochs)
        self.control = None
        self.delta_c = None
        self.delta_y = None
        self.server_control = None
    
    def _receive_model(self) -> None:
        model, server_control = self.channel.receive(self, self.server, msg_type="model").payload
        if self.model is None:
            self.model = deepcopy(model)
            self.control = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
            self.delta_y = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
            self.delta_c = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
        else:
            self.model.load_state_dict(model.state_dict())
        self.server_control = server_control
    
    def local_train(self, override_local_epochs: int=0):
        epochs = override_local_epochs if override_local_epochs else self.local_epochs
        self._receive_model()
        server_model = deepcopy(self.model)
        self.model.train()
        if self.optimizer is None:
            self.optimizer, self.scheduler = self.optimizer_cfg(self.model)
        for _ in range(epochs):
            loss = None
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_hat = self.model(X)
                loss = self.loss_fn(y_hat, y)
                loss.backward()
                self.optimizer.step(self.server_control, self.control)
            self.scheduler.step()
        
        #TODO: get only the trainable parameters
        for local_model, server_model, delta_y in zip(self.model.parameters(), server_model.parameters(), self.delta_y):
            delta_y.data = local_model.data.detach() - server_model.data.detach()
        
        new_controls = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
        coeff = 1. / (self.local_epochs * len(self.train_set) * self.scheduler.get_last_lr()[0])
        for local_control, server_control, new_control, delta_y in zip(self.control, self.server_control, new_controls, self.delta_y):
            new_control.data = local_control.data - server_control.data - delta_y.data * coeff

        for local_control, new_control, delta_c in zip(self.control, new_controls, self.delta_c):
            delta_c.data = new_control.data - local_control.data
            local_control.data = new_control.data
        
        self.channel.send(Message((self.delta_y, self.delta_c), "model", self), self.server)


class ScaffoldServer(Server):
    def __init__(self,
                 model: Module,
                 clients: Iterable[Client],
                 global_step: float=1.,
                 eligibility_percentage: float=0.5):
        super().__init__(model, clients, eligibility_percentage)
        self.control = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
        self.global_step = global_step
    
    def _broadcast_model(self, eligible: Iterable[Client]) -> None:
        self.channel.broadcast(Message((self.model, self.control), "model", self), eligible)

    def aggregate(self, eligible: Iterable[Client]) -> None:
        with torch.no_grad():
            delta_y = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
            delta_c = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]

            for client in eligible:
                cl_delta_y, cl_delta_c = self.channel.receive(self, client, "model").payload
                for client_delta_c, client_delta_y, server_delta_c, server_delta_y in zip(cl_delta_c, cl_delta_y, delta_c, delta_y):
                    server_delta_y.data = server_delta_y.data + client_delta_y.data
                    server_delta_c.data = server_delta_c.data + client_delta_c.data
                
            for server_delta_c, server_delta_y in zip(delta_c, delta_y):
                server_delta_y.data = server_delta_y.data / len(eligible) #* (eligible[i].n_examples / tot_examples)
                server_delta_c.data = server_delta_c.data / self.n_clients

            for param, server_control, server_delta_y, server_delta_c in zip(self.model.parameters(), self.control, delta_y, delta_c):
                param.data = param.data + self.global_step * server_delta_y
                server_control.data = server_control.data + server_delta_c.data


class SCAFFOLD(CentralizedFL):
    """SCAFFOLD Federated Learning Environment.

    https://arxiv.org/pdf/1910.06378.pdf

    Parameters
    ----------
    n_clients : int
        Number of clients in the FL environment.
    n_rounds : int
        Number of communication rounds.
    n_epochs : int
        Number of epochs per communication round.
    optimizer_cfg : OptimizerConfigurator
        Optimizer configurator for the clients.
    model : torch.nn.Module
        Model to be trained.
    loss_fn : Callable
        Loss function.
    eligibility_percentage : float, optional
        Percentage of clients to be selected for each communication round, by default 0.5.
    """
    def __init__(self,
                 n_clients: int,
                 n_rounds: int, 
                 n_epochs: int,
                 optimizer_cfg: OptimizerConfigurator,
                 global_step: float,
                 model: Module,
                 loss_fn: Callable,
                 eligibility_percentage: float=0.5):
        
        super().__init__(n_clients,
                         n_rounds,
                         n_epochs,
                         model, 
                         optimizer_cfg, 
                         loss_fn,
                         eligibility_percentage)
        self.global_step = global_step
    
    def init_clients(self, data_splitter: DataSplitter, **kwargs):
        assert data_splitter.n_clients == self.n_clients, "Number of clients in data splitter and the FL environment must be the same"
        self.clients = [ScaffoldClient(train_set=data_splitter.client_train_loader[i], 
                                        optimizer_cfg=self.optimizer_cfg, 
                                        loss_fn=self.loss_fn, 
                                        validation_set=data_splitter.client_test_loader[i],
                                        local_epochs=self.n_epochs) for i in range(self.n_clients)]
    
    def init_server(self, **kwargs):
        self.server = ScaffoldServer(self.model, 
                                     self.clients, 
                                     self.global_step, 
                                     self.eligibility_percentage)
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(C={self.n_clients},R={self.n_rounds},E={self.n_epochs}," + \
               f"G={self.global_step},P={self.eligibility_percentage},{self.optimizer_cfg})"
