from collections import OrderedDict
from typing import Callable, Iterable

import torch
from torch.nn import Module, MSELoss
from torch.utils.data import DataLoader, TensorDataset

import sys; sys.path.append(".")
from fl_bench.client import Client
from fl_bench.server import Server
from fl_bench.data import DataSplitter, FastTensorDataLoader
from fl_bench.utils import OptimizerConfigurator
from fl_bench.algorithms import CentralizedFL


class FLHalfClient(Client):
    def __init__(self,
                 dataset: FastTensorDataLoader,
                 private_layers: Iterable,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: Callable, # CHECK ME
                 local_epochs: int=3,
                 seed: int=42):
        super().__init__(dataset, optimizer_cfg, loss_fn, local_epochs, seed)
        self.private_layers = private_layers
    
    def _generate_fake_examples(self):
        shape = list(self.dataset.tensors[0].shape)
        fake_data = torch.rand(shape)
        fake_targets = self.model.forward_(fake_data)
        return fake_data, fake_targets
    
    def send(self):
        return super().send(), self._generate_fake_examples()


class FLHalfServer(Server):
    def __init__(self,
                 model: Module,
                 clients: Iterable[Client],
                 private_layers: Iterable,
                 n_epochs: int,
                 batch_size: int,
                 optimizer_cfg: OptimizerConfigurator,
                 global_step: float=.05,
                 elegibility_percentage: float=0.5, 
                 seed: int=42):
        super().__init__(model, clients, elegibility_percentage, seed)
        self.control = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
        self.global_step = global_step
        self.private_layers = private_layers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer_cfg(self.model)
    
    def _private_train(self, clients_fake_x, clients_fake_y):
        train_loader = FastTensorDataLoader(clients_fake_x, 
                                            clients_fake_y, 
                                            batch_size=self.batch_size, 
                                            shuffle=True)
        loss_fn = MSELoss()
        for _ in range(self.n_epochs):
            loss = None
            for _, (X, y) in enumerate(train_loader):
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_hat = self.model.forward_(X)
                loss = loss_fn(y_hat, y)
                loss.backward(retain_graph=True)
                self.optimizer.step()

    def aggregate(self, eligible: Iterable[Client]) -> None:
        avg_model_sd = OrderedDict()
        clients_sd = []
        clients_fake_x = []
        clients_fake_y = []
        for i in range(len(eligible)):
            client_model, (client_fake_x, client_fake_y) = eligible[i].send()
            clients_sd.append(client_model.state_dict())
            clients_fake_x.append(client_fake_x)
            clients_fake_y.append(client_fake_y)
        
        self._private_train(torch.cat(clients_fake_x, 0), torch.cat(clients_fake_y, 0))

        global_model_dict = self.model.state_dict()
        for key in global_model_dict.keys():
            if key.split(".")[0] in self.private_layers:
                avg_model_sd[key] = global_model_dict[key]
                continue
            for i, client_sd in enumerate(clients_sd):
                if key not in avg_model_sd:
                    avg_model_sd[key] = client_sd[key]
                else:
                    avg_model_sd[key] += client_sd[key]
            avg_model_sd[key] /= len(eligible)
        self.model.load_state_dict(avg_model_sd)


class FLHalf(CentralizedFL):
    def __init__(self,
                 n_clients: int,
                 n_rounds: int, 
                 client_n_epochs: int, 
                 server_n_epochs: int,
                 client_batch_size: int, 
                 server_batch_size: int,
                 client_optimizer_cfg: OptimizerConfigurator, 
                 server_optimizer_cfg: OptimizerConfigurator, 
                 model: Module, 
                 private_layers: Iterable,
                 loss_fn: Callable, 
                 elegibility_percentage: float=0.5,
                 seed: int=42):
        
        super().__init__(n_clients,
                         n_rounds,
                         client_n_epochs,
                         client_batch_size,
                         model, 
                         client_optimizer_cfg, 
                         loss_fn,
                         elegibility_percentage,
                         seed)
        self.private_layers = private_layers
        self.server_n_epochs = server_n_epochs
        self.server_batch_size = server_batch_size
        self.server_optimizer_cfg = server_optimizer_cfg

    def init_parties(self, data_splitter: DataSplitter, callback: Callable=None):
        assert data_splitter.n_clients == self.n_clients, "Number of clients in data splitter and the FL environment must be the same"
        self.clients = [FLHalfClient (dataset=data_splitter.client_loader[i], 
                                      private_layers=self.private_layers,
                                      optimizer_cfg=self.optimizer_cfg, 
                                      loss_fn=self.loss_fn, 
                                      local_epochs=self.n_epochs,
                                      seed=self.seed) for i in range(self.n_clients)]

        self.server = FLHalfServer(self.model,
                                   self.clients, 
                                   private_layers=self.private_layers, 
                                   n_epochs=self.server_n_epochs,
                                   batch_size=self.server_batch_size,
                                   optimizer_cfg=self.server_optimizer_cfg,
                                   elegibility_percentage=self.elegibility_percentage, 
                                   seed=self.seed)
        self.server.register_callback(callback)
