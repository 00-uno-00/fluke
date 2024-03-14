import sys
import torch; sys.path.append(".")

import typer

import rich
from rich.progress import track
from rich.panel import Panel
from rich.pretty import Pretty

from fl_bench import GlobalSettings
from fl_bench.data import DataSplitter, FastTensorDataLoader
from fl_bench.utils import Configuration, OptimizerConfigurator, get_loss, get_model
from fl_bench.evaluation import ClassificationEval, ClassificationSklearnEval
from fl_bench.algorithms import FedAlgorithmsEnum
from fl_bench.algorithms.boost import FedAdaboostAlgorithmsEnum

app = typer.Typer()

# CONST
CONFIG_FNAME = "configs/exp_settings.json"


@app.command()
def run_centralized(alg_cfg: str = typer.Argument(..., help='Config file for the algorithm to run'),
                    epochs: int = typer.Option(0, help='Number of epochs to run')):

    cfg = Configuration(CONFIG_FNAME, alg_cfg)
    GlobalSettings().set_seed(cfg.exp.seed)
    GlobalSettings().set_device(cfg.exp.device)
    data_container = cfg.data.dataset.klass()(**cfg.data.dataset.exclude('name'))

    device = GlobalSettings().get_device()

    train_loader = FastTensorDataLoader(*data_container.train, 
                                             batch_size=cfg.method.hyperparameters.client.batch_size, 
                                             shuffle=True)
    test_loader = FastTensorDataLoader(*data_container.test,
                                            batch_size=1,#cfg.method.hyperparameters.client.batch_size, 
                                            shuffle=False)

    model = get_model(mname=cfg.method.hyperparameters.model)#, **cfg.method.hyperparameters.net_args)
    optimizer_cfg = OptimizerConfigurator(torch.optim.SGD, 
                                              **cfg.method.hyperparameters.client.optimizer.exclude('scheduler_kwargs'),
                                              scheduler_kwargs=cfg.method.hyperparameters.client.optimizer.scheduler_kwargs)
    optimizer, scheduler = optimizer_cfg(model)
    criterion = get_loss(cfg.method.hyperparameters.client.loss)

    evaluator = ClassificationEval(criterion, data_container.num_classes(), cfg.exp.average, device=device)

    # log = cfg.log.logger.logger(evaluator,
    #                             eval_every=cfg.log.eval_every,
    #                             name=str(cfg),
    #                             **cfg.log.wandb_params)
    
    model.to(device)
    epochs = epochs if epochs > 0 else int(max(1, cfg.protocol.n_rounds * cfg.protocol.eligible_perc))
    for e in range(epochs):
        model.train()
        print(f"Epoch {e+1}")
        loss = None
        for _, (X, y) in track(enumerate(train_loader), total=train_loader.n_batches):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = model(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
        print(evaluator.evaluate(model, test_loader))
        print()
    model.to("cpu")

@app.command()
def run(alg_cfg: str = typer.Argument(..., help='Config file for the algorithm to run')):

    cfg = Configuration(CONFIG_FNAME, alg_cfg)
    GlobalSettings().set_seed(cfg.exp.seed) 
    GlobalSettings().set_device(cfg.exp.device)
    data_splitter = DataSplitter.from_config(cfg.data)

    fl_algo_builder = FedAlgorithmsEnum(cfg.method.name)
    fl_algo = fl_algo_builder.algorithm()(cfg.protocol.n_clients, data_splitter, cfg.method.hyperparameters)

    log = cfg.logger.name.logger(ClassificationEval(fl_algo.loss, 
                                                   data_splitter.num_classes(),
                                                   cfg.exp.average,
                                                   GlobalSettings().get_device()), 
                                eval_every=cfg.logger.eval_every,
                                name=str(cfg),
                                **cfg.logger.exclude('name', 'eval_every'))
    log.init(**cfg)
    fl_algo.set_callbacks(log)
    
    # if cfg.exp.checkpoint.load:
    #     fl_algo.load_checkpoint(cfg.exp.checkpoint.path)
    
    # if cfg.exp.checkpoint.save:
    #     fl_algo.activate_checkpoint(cfg.exp.checkpoint.path)

    rich.print(Panel(Pretty(fl_algo), title=f"FL algorithm"))
    # GlobalSettings().set_workers(8)
    fl_algo.run(cfg.protocol.n_rounds, cfg.protocol.eligible_perc)
    # log.save(f'./log/{fl_algo}_{cfg.dataset.value}_{cfg.distribution.value}.json')


@app.command()
def run_boost(alg_cfg: str = typer.Argument(..., help='Config file for the algorithm to run')):

    cfg = Configuration(CONFIG_FNAME, alg_cfg)
    GlobalSettings().set_seed(cfg.exp.seed) 
    GlobalSettings().set_device(cfg.exp.device)
    data_splitter = DataSplitter.from_config(cfg.data)
    
    log = cfg.log.logger.logger(ClassificationSklearnEval("macro"), 
                                name=str(cfg),
                                **cfg.log.wandb_params)
    log.init(**cfg)

    fl_algo_builder = FedAdaboostAlgorithmsEnum(cfg.method.name)
    fl_algo = fl_algo_builder.algorithm()(cfg.protocol.n_clients, data_splitter, cfg.method.hyperparameters)
    fl_algo.set_callbacks(log)

    rich.print(Panel(Pretty(fl_algo), title=f"FL algorithm"))

    fl_algo.run(cfg.protocol.n_rounds, cfg.protocol.eligible_perc)
    # log.save(f'./log/{fl_algo}_{cfg.dataset.value}_{cfg.distribution.value}.json')


@app.command()
def validate(alg_cfg: str = typer.Argument(..., help='Config file for the algorithm to run')):
    cfg = Configuration(CONFIG_FNAME, alg_cfg)
    cfg._validate()
    rich.print(Panel(Pretty(cfg, expand_all=True), title=f"Configuration"))
    



@app.callback()
def main(config: str=typer.Option(CONFIG_FNAME, help="Configuration file")):
    global CONFIG_FNAME
    CONFIG_FNAME = config


if __name__ == '__main__':
    app()