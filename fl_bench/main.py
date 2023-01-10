from copy import deepcopy
import torch
import glob
import torch.nn as nn
import json

import sys; sys.path.append(".")
from fl_bench import GlobalSettings
from net import *
from data import DataSplitter, Distribution, FastTensorDataLoader, DatasetsEnum
from utils import WandBLog, plot_comparison
from utils import OptimizerConfigurator, Log, set_seed, load_defaults, get_loss
from evaluation import ClassificationEval
from algorithms import FedAlgorithmsEnum


from enum import Enum

from rich.console import Console
import typer
app = typer.Typer()


console = Console()

GlobalSettings().auto_device()
DEVICE = GlobalSettings().get_device()


DEFAULTS = load_defaults()



@app.command()
def run(algorithm: FedAlgorithmsEnum = typer.Option(DEFAULTS["method"]["name"], help='Algorithm to run'),
        dataset: DatasetsEnum = typer.Option(DEFAULTS["dataset"], help='Dataset'),
        n_clients: int = typer.Option(DEFAULTS["n_clients"], help='Number of clients'),
        n_rounds: int = typer.Option(DEFAULTS["n_rounds"], help='Number of rounds'),
        n_epochs: int = typer.Option(DEFAULTS["n_epochs"], help='Number of epochs'),
        batch_size: int = typer.Option(DEFAULTS["batch_size"], help='Batch size'),
        elegibility_percentage: float = typer.Option(DEFAULTS["eligibility_percentage"], help='Elegibility percentage'),
        distribution: Distribution = typer.Option(DEFAULTS["distribution"], help='Data distribution'),
        seed: int = typer.Option(DEFAULTS["seed"], help='Seed')):
    
    set_seed(seed) #Reproducibility

    print("Running configuration:")
    options = deepcopy(locals())
    console.log(options)
    print()

    MODEL = MLP().to(DEVICE)
    loss = get_loss(DEFAULTS["loss"])

    data_container = dataset.klass()()

    data_splitter = DataSplitter(*data_container.train,
                                 n_clients=n_clients, 
                                 distribution=distribution, 
                                 batch_size=batch_size,
                                 validation_split=.1,
                                 sampling_perc=1.)

    test_loader = FastTensorDataLoader(*data_container.test,
                                       batch_size=100, 
                                       shuffle=False)

    logger = Log(ClassificationEval(test_loader, loss, data_container.num_classes, "macro"))
    # logger = WandBLog(ClassificationEval(test_loader, loss), 
    #                   project="fl-bench",
    #                   entity="mlgroup",
    #                   name=f"{algorithm}_{dataset}_{IIDNESS_MAP[Distribution(distribution)]}", 
    #                   config=options)

    fl_algo = algorithm.algorithm()( n_clients=n_clients,
                                    n_rounds=n_rounds, 
                                    n_epochs=n_epochs, 
                                    elegibility_percentage=elegibility_percentage,
                                    model=MODEL,
                                    loss_fn = loss,
                                    optimizer_cfg=OptimizerConfigurator(
                                        algorithm.optimizer(), 
                                        lr=DEFAULTS["method"]["optimizer_parameters"]["lr"], 
                                        scheduler_kwargs=DEFAULTS["method"]["optimizer_parameters"]["scheduler_kwargs"]),
                                    **DEFAULTS["method"]["hyperparameters"])

    
    print("FL algorithm: ") 
    console.log(fl_algo)
    print()
    
    fl_algo.init_parties(data_splitter, callback=logger)
    fl_algo.run()
    logger.save(f'./log/{fl_algo}_{dataset.value}_{distribution}.json')


@app.command()
def compare(dataset: str=typer.Option('mnist', help='Dataset'),
            n_clients: int=typer.Option(100, help='Number of clients'),
            n_rounds: int=typer.Option(100, help='Number of rounds'),
            distribution: int=typer.Option(Distribution.IID.value, help='Data distribution'),
            show_loss: bool=typer.Option(True, help='Show loss graph'),
            local: bool=typer.Option(False, help='Compare client-side results')):

    paths = glob.glob(f'./log/*C={n_clients},R={n_rounds},*_{dataset}_{distribution}.json')
    plot_comparison(*paths, local=local, show_loss=show_loss)

# compare('./log/flhalf_noniid_dir.json', './log/fedavg_noniid_dir.json', show_loss=True) 


if __name__ == '__main__':
    app()