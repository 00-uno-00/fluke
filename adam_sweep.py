import os
import sys
import argparse

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
    cfg = None

    @staticmethod
    def Train(cfg):
        dataset = Datasets.get("mnist", path="./data")
        
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

    @staticmethod
    def parse_args():
        argparser = argparse.ArgumentParser(description="Run ADAM")
        argparser.add_argument("--exp", type=str, default="./configs/exp_adam_bench.yaml")
        argparser.add_argument("--fed", type=str, default="./configs/fedadam_bench.yaml")
        argparser.add_argument("--tau", type=float, default=0.0001)
        argparser.add_argument("--lr", type=float, default=0.1)
        argparser.add_argument("--beta1", type=float, default=0.9)
        argparser.add_argument("--beta2", type=float, default=0.999)
        args = argparser.parse_args()
        ADAM.cfg = Configuration(args.exp, args.fed)
        ADAM.cfg.method.hyperparameters.server.tau = args.tau
        ADAM.cfg.method.hyperparameters.client.optimizer.lr = args.lr
        ADAM.cfg.method.hyperparameters.server.beta1 = args.beta1
        ADAM.cfg.method.hyperparameters.server.beta2 = args.beta2
        return 

if __name__ == '__main__':
    ADAM.parse_args()
    ADAM.Train(ADAM.cfg)