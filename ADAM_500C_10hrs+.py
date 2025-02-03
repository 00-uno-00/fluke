import os
import sys

# Add the path to the fluke library
fluke_path = os.path.abspath(os.path.join(os.path.dirname(__file__), './fluke'))
print(fluke_path)
sys.path.append(fluke_path)

from fluke.data.datasets import Datasets
from fluke.utils import Configuration, get_class_from_qualified_name
from fluke.utils.log import get_logger
from fluke.run import GlobalSettings
from fluke.evaluation import ClassificationEval
import rich
from rich.panel import Panel
from rich.pretty import Pretty
import random
import torch
from torchvision.transforms import v2
from fluke.data import DataSplitter
from fluke import DDict

class ADAM:
    
    transforms_train = v2.Compose([
        v2.RandomResizedCrop(size=(24, 24), antialias=True),
        v2.RandomHorizontalFlip(p=random.uniform(0, 1)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.49139968, 0.48215827 ,0.44653124], std=[0.24703233, 0.24348505, 0.26158768]),
    ])

    transforms_test = v2.Compose([
        v2.CenterCrop(size=(24, 24)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.49139968, 0.48215827 ,0.44653124], std=[0.24703233, 0.24348505, 0.26158768]),
    ])
    dataset = Datasets.get("cifar10", path="./data", transforms_test=transforms_test, transforms_train=transforms_train)
    

    cfg = Configuration("./configs/exp_adam.yaml", "./configs/fedadam.yaml")
    data_splitter = DataSplitter(dataset=dataset, distribution=cfg.data.distribution.name, dist_args=DDict(cfg.data.distribution.class_per_client))

    evaluator = ClassificationEval(cfg.eval.eval_every, n_classes=10)

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
