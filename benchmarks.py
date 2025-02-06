import os
import sys
import argparse

# Add the path to the fluke library
fluke_path = os.path.abspath(os.path.join(os.path.dirname(__file__), './fluke'))
print(fluke_path)
sys.path.append(fluke_path)

from fluke.data.datasets import *
from fluke.utils import get_class_from_qualified_name, Configuration
from fluke.utils.log import get_logger
from fluke.run import GlobalSettings
from fluke.evaluation import ClassificationEval
import rich
from rich.panel import Panel
from rich.pretty import Pretty
from fluke.utils.log import *
from fluke.data import *
from fluke import DDict

# Define the algorithms
algorithms = {
    "avg": "fedavg",
    "prox": "fedprox",
    "scaffold": "scaffold",
    "dyn": "feddyn",
    "moon": "moon",
    "lc": "fedlc",
    "rs": "fedrs",
    "kafe": "fedkafe",
    "avgm": "fedavgm"
  }  # Add more algorithms here

# Define the settings
alg_settings = [
    {"batch_size": 10, 
     "local_epochs": 10, 
     "loss": "CrossEntropyLoss", 
     "lr": 0.01,
     "gamma": 1,
     "step_size": 1,
     "model": "MNIST_2NN",
     "weighted": "true"
     },
     {"batch_size": 32, 
     "local_epochs": 10, 
     "loss": "CrossEntropyLoss", 
     "lr": 0.01,
     "gamma": 1,
     "step_size": 1,
     "model": "MNIST_2NN",
     "weighted": "true"
     },    
     {"batch_size": 10, 
     "local_epochs": 10, 
     "loss": "CrossEntropyLoss", 
     "lr": 0.01,
     "gamma": 1,
     "step_size": 1,
     "model": "ResNet18",
     "weighted": "true"
     },
     {"batch_size": 32, 
     "local_epochs": 10, 
     "loss": "CrossEntropyLoss", 
     "lr": 0.01,
     "gamma": 1,
     "step_size": 1,
     "model": "ResNet18",
     "weighted": "true"
     }
]

exp_settings = [
    {"config_name": "Simple",
     "dataset_name": "mnist",
     "channel_dim": 1,
     "classes": 10,
     "distribution_name": "iid", 
     "sampling_perc": 1, 
     "client_split": 0.2, 
     "keep_test": "true",  
     "server_test": "true", 
     "server_split": 0.2, 
     "uniform_test": "false",
     "device": "cuda",
     "exp_seed": 42, 
     "exp_inmemory": "true",
     "eval_every": 1,
     "eligible_perc": 0.2, 
     "n_clients": 50, 
     "n_rounds": 100},

     {"config_name": "Complex",
     "dataset_name": "cifar10",
     "channel_dim": 3,
     "classes": 10,
     "distribution_name": "iid", 
     "sampling_perc": 1, 
     "client_split": 0.2, 
     "keep_test": "true",  
     "server_test": "true", 
     "server_split": 0.2, 
     "uniform_test": "false",
     "device": "cuda",
     "exp_seed": 42, 
     "exp_inmemory": "true",
     "eval_every": 1,
     "eligible_perc": 0.2, 
     "n_clients": 50, 
     "n_rounds": 200}
]

def main():
    for algorithm in algorithms:
        for exp_setting in exp_settings:
            for alg_setting in alg_settings :
                dataset = Datasets.get(exp_setting['dataset_name'], path="./data", channel_dim=int(exp_setting['channel_dim'] if alg_setting['model'] != "ResNet18" else None))

                cfg = Configuration(f"./configs/exp_{algorithm}.yaml", f"./configs/fed{algorithm}.yaml")
                dist_args=DDict(
                    dataset=dataset,
                    distribution=exp_setting['distribution_name'],
                    sampling_perc=exp_setting['sampling_perc'],
                    client_split=exp_setting['client_split'],
                    keep_test=exp_setting['keep_test'],
                    server_test=exp_setting['server_test'],
                    server_split=exp_setting['server_split'],
                    uniform_test=exp_setting['uniform_test']
                    )

                #data_splitter = DataSplitter(dataset=dataset, distribution=exp_setting['distribution_name'], dist_args=dist_args)
                data_splitter = DataSplitter(**dist_args)

                evaluator = ClassificationEval(exp_setting['eval_every'], n_classes=exp_setting['classes'])

                settings = GlobalSettings()
                settings.set_device(exp_setting['device'])
                settings.set_seed(exp_setting['exp_seed'])
                settings.set_evaluator(evaluator=evaluator)

                # Overwrite the hyperparameters
                cfg.method.hyperparameters.client.batch_size=int(alg_setting['batch_size'])
                cfg.method.hyperparameters.client.local_epochs=int(alg_setting['local_epochs'])
                cfg.method.hyperparameters.client.loss=str(alg_setting['loss'])
                cfg.method.hyperparameters.client.optimizer.lr=float(alg_setting['lr'])
                cfg.method.hyperparameters.client.scheduler.gamma=float(alg_setting['gamma'])
                cfg.method.hyperparameters.client.scheduler.step_size=int(alg_setting['step_size'])

                cfg.method.hyperparameters.model=str(alg_setting['model'])
                cfg.method.hyperparameters.server.weighted=bool(alg_setting['weighted'])

                cfg.logger.tags = cfg.logger.tags + [str(exp_setting['config_name'])]
                cfg.logger.project = "fluke_benchmarks_final"

                fl_algo_class = get_class_from_qualified_name(cfg.method.name)
                fl_algo = fl_algo_class(n_clients=exp_setting['n_clients'], data_splitter=data_splitter, hyper_params=cfg.method.hyperparameters)
                
                log = get_logger(cfg.logger.name, name=str(cfg), **cfg.logger.exclude('name'))
                log.init(**cfg)
                fl_algo.set_callbacks(log)
                rich.print(Panel(Pretty(fl_algo), title="FL algorithm"))

                fl_algo.run(exp_setting['n_rounds'], exp_setting['eligible_perc'])
                print(f"Experiment {exp_setting['config_name']} with {algorithm} and {alg_setting['model']} is done.")



if __name__ == "__main__":
    main()