from typing import Callable, Union, Any, Iterable

import torch
from enum import Enum

from fl_bench.client import Client
from fl_bench.server import Server
from fl_bench.data import DataSplitter, FastTensorDataLoader
from fl_bench.utils import DDict, OptimizerConfigurator, get_loss, get_model


class CentralizedFL():
    """Generic Centralized Federated Learning algorithm.

    This class is a generic implementation of a centralized federated learning algorithm. 

    Args:
        n_clients (int): Number of clients.
        data_splitter (DataSplitter): Data splitter object.
        hyperparameters (DDict): Hyperparameters for the algorithm.
    """
    def __init__(self, 
                 n_clients: int,
                 data_splitter: DataSplitter, 
                 hyperparameters: DDict):
        self.hyperparameters = hyperparameters
        self.n_clients = n_clients
        (clients_tr_data, clients_te_data), server_data = data_splitter.assign(n_clients, 
                                                                               hyperparameters.client.batch_size)
        # Federated model
        model = self.init_model(
                model_name=hyperparameters.model,
                **hyperparameters.net_args if 'net_args' in hyperparameters else {}
            )

        self.init_clients(clients_tr_data, clients_te_data, hyperparameters.client)
        self.init_server(model, server_data, hyperparameters.server)

    def get_optimizer_class(self) -> torch.optim.Optimizer:
        return torch.optim.SGD

    def get_client_class(self) -> Client:
        return Client

    def get_server_class(self) -> Server:
        return Server

    def init_model(self, model_name, **args):
        return get_model(mname=model_name, **args)
        
    def init_clients(self, 
                     clients_tr_data: list[FastTensorDataLoader], 
                     clients_te_data: list[FastTensorDataLoader], 
                     config: DDict) -> None:
        optimizer_cfg = OptimizerConfigurator(self.get_optimizer_class(), 
                                              **config.optimizer.exclude('scheduler_kwargs'),
                                              scheduler_kwargs=config.optimizer.scheduler_kwargs)
        self.loss = get_loss(config.loss)
        self.clients = [
            self.get_client_class()(
                index=i,
                train_set=clients_tr_data[i],  
                test_set=clients_te_data[i],
                optimizer_cfg=optimizer_cfg, 
                loss_fn=self.loss, 
                **config.exclude('optimizer', 'loss', 'batch_size')
            ) 
            for i in range(self.n_clients)]

    def init_server(self, model: Any, data: FastTensorDataLoader, config: DDict):
        self.server = self.get_server_class()(model, data, self.clients, **config)
    
    def set_callbacks(self, callbacks: Union[Callable, Iterable[Callable]]):
        self.server.attach(callbacks)
        self.server.channel.attach(callbacks)
        
    def run(self, n_rounds: int, eligible_perc: float):
        self.server.fit(n_rounds=n_rounds, eligible_perc=eligible_perc)
    
    def __str__(self) -> str:
        algo_hp = ",\n\t".join([f"{h}={str(v)}" for h,v in self.hyperparameters.items() if h not in ['client', 'server']])
        algo_hp = f"\n\t{algo_hp}," if algo_hp else ""
        return f"{self.__class__.__name__}({algo_hp}\n\t{self.clients[0]},\n\t{self.server}\n)"
    
    def __repr__(self) -> str:
        return self.__str__()


class PersonalizedFL(CentralizedFL):
    
    def init_clients(self, 
                     clients_tr_data: list[FastTensorDataLoader], 
                     clients_te_data: list[FastTensorDataLoader], 
                     config: DDict) -> None:

        model = get_model(mname=config.model)
        optimizer_cfg = OptimizerConfigurator(self.get_optimizer_class(), 
                                              **config.optimizer.exclude('scheduler_kwargs'),
                                              scheduler_kwargs=config.optimizer.scheduler_kwargs)
        self.loss = get_loss(config.loss)
        self.clients = [
            self.get_client_class()(
                index=i,
                model=model,
                train_set=clients_tr_data[i],  
                test_set=clients_te_data[i],
                optimizer_cfg=optimizer_cfg, 
                loss_fn=self.loss, 
                **config.exclude('optimizer', 'loss', 'batch_size', 'model')
            ) 
            for i in range(self.n_clients)]


# FEDERATED LEARNING ALGORITHMS
from .fedavg import FedAVG
from .fedavgm import FedAVGM
from .fedsgd import FedSGD
from .fedprox import FedProx
from .scaffold import SCAFFOLD
from .flhalf import FLHalf
from .fedbn import FedBN
from .fedopt import FedOpt
from .moon import MOON
from .fednova import FedNova
from .fedexp import FedExP
from .pfedme import PFedMe
from .feddisel import FedDisel
from .feddyn import FedDyn
from .fedper import FedPer
from .fedrep import FedRep
from .ditto import Ditto
from .apfl import APFL
from .superfed import SuPerFed


class FedAlgorithmsEnum(Enum):
    FEDAVG = 'fedavg'
    FEDAGM = 'fedavgm'
    FEDSGD = 'fedsgd'
    FEDPROX = 'fedprox'
    SCAFFOLD = 'scaffold'
    FLHALF = 'flhalf'
    FEDBN = 'fedbn'
    FEDOPT = 'fedopt'
    MOON = 'moon'
    FEDNOVA = 'fednova'
    FEDEXP = 'fedexp'
    PEFEDME = 'pfedme'
    FEDDYN = 'feddyn'
    FEDDISEL = 'feddisel'
    FEDPER = 'fedper'
    FEDREP = 'fedrep'
    DITTO = 'ditto'
    APFL = 'apfl'
    SUPERFED = 'superfed'

    @classmethod
    def contains(cls, member: object) -> bool:
        if isinstance(member, str):
            return member in cls._value2member_map_.keys()
        elif isinstance(member, FedAlgorithmsEnum):
            return member.value in cls._member_names_
        
    def algorithm(self):
        algos = {
            'fedavg': FedAVG,
            'fedavgm': FedAVGM,
            'fedsgd': FedSGD,
            'fedprox': FedProx,
            'scaffold': SCAFFOLD,
            'flhalf': FLHalf,
            'fedbn': FedBN,
            'fedopt': FedOpt,
            'moon': MOON,
            'fednova': FedNova,
            'fedexp': FedExP,
            'pfedme': PFedMe,
            'feddyn': FedDyn,
            'feddisel': FedDisel,
            'fedper': FedPer,
            'fedrep': FedRep,
            'ditto': Ditto,
            'apfl': APFL,
            'superfed': SuPerFed
        }

        return algos[self.value]
