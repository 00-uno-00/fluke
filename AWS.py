import os
import sys
sys.path.append(os.path.abspath('.'))

from fluke.data.datasets import *
from fluke.data import *
from fluke.utils import DDict, Configuration, get_class_from_qualified_name
from fluke.utils.log import get_logger
from fluke.run import GlobalSettings
from fluke.evaluation import ClassificationEval
import rich
from rich.panel import Panel
from rich.pretty import Pretty

class AwS:
    
    dataset = Datasets.get("cifar10", path="./data")

    cfg = Configuration("./configs/exp_aws.yaml", "./configs/fedaws.yaml")

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
