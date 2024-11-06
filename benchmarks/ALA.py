from fluke.data.datasets import Datasets
from fluke import DDict
from fluke.utils import Configuration 
from fluke.evaluation import ClassificationEval
from fluke import GlobalSettings
from fluke.data import DataSplitter

from typing import Any, Iterable, Optional, Union
import torch

from typing import Iterable
import rich
from fluke.data import FastDataLoader  # NOQA

from fluke.utils import Configuration
from rich.panel import Panel
from rich.pretty import Pretty
from rich.progress import track

from fluke import GlobalSettings  # NOQA
from fluke.data import DataSplitter, FastDataLoader  # NOQA
from fluke.data.datasets import Datasets  # NOQA
from fluke.evaluation import ClassificationEval  # NOQA
from fluke.utils import (Configuration, OptimizerConfigurator,  # NOQA
                    get_class_from_qualified_name, get_loss, get_model)
from fluke.utils.log import get_logger  # NOQA

class BestLocalAccuracy(ClassificationEval):
    def __init__(self, eval_every: int, n_classes: int):
        super().__init__(eval_every=eval_every, n_classes=n_classes)
        self.best_acc = 0
        self.accuracies = []
        
    def evaluate(self,
                 round: int,
                 model: torch.nn.Module,
                 eval_data_loader: Union[FastDataLoader, Iterable[FastDataLoader]],
                 loss_fn: Optional[torch.nn.Module] = None,
                 device: torch.device = torch.device("cpu"),
                 weights: torch.tensor = None) -> dict:
        
        model.eval()
        model.to(device)
        correct = 0
        total = 0
        with torch.no_grad():
            for data in eval_data_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        accuracy = accuracy/100
        self.accuracies.append(accuracy)
        
        if accuracy > self.best_acc:
            self.best_acc = accuracy
        
        model.to("cpu")
        metrics = super().evaluate(round, model, eval_data_loader, loss_fn, device)
        metrics["best_accuracy"] = self.best_acc

        return metrics
    
class Ala:
    
    dataset = Datasets.get("cifar10", path="../data")

    cfg = Configuration("../configs/exp_ala.yaml", "../configs/fedala.yaml")

    data_splitter = DataSplitter(dataset=dataset, distribution=cfg.data.distribution.name, dist_args=DDict(cfg.data.distribution.beta, cfg.data.distribution.balanced))

    evaluator = BestLocalAccuracy(eval_every=cfg.eval.eval_every, n_classes=10)

    GlobalSettings().set_device(cfg.exp.device)
    GlobalSettings().set_seed(cfg.exp.seed)
    GlobalSettings().set_eval_cfg(cfg.eval)
    GlobalSettings().set_evaluator(evaluator=evaluator)

    fl_algo_class = get_class_from_qualified_name(cfg.method.name)
    fl_algo = fl_algo_class(cfg.protocol.n_clients,
                                data_splitter,
                                cfg.method.hyperparameters)

    log = get_logger(cfg.logger.name, name=str(cfg), **cfg.logger.exclude('name'))
    log.init(**cfg)
    fl_algo.set_callbacks(log)
    rich.print(Panel(Pretty(fl_algo), title="FL algorithm"))

    fl_algo.run(cfg.protocol.n_rounds, cfg.protocol.eligible_perc)
