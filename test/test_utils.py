from torch.nn import Linear, CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import pytest
import json
import tempfile
import torch
import sys
sys.path.append(".")
sys.path.append("..")


from fl_bench.nets import MNIST_2NN, VGG9, Shakespeare_LSTM  # NOQA
from fl_bench.comm import Message  # NOQA
from fl_bench.client import Client  # NOQA
from fl_bench.utils import (OptimizerConfigurator, import_module_from_str,  # NOQA
           get_class_from_str, get_model, get_class_from_qualified_name,  # NOQA
           get_full_classname, get_loss, get_scheduler, clear_cache, Configuration,  # NOQA
           Log, WandBLog)  # NOQA

from fl_bench.utils.model import (merge_models, diff_model, mix_networks,   # NOQA
                                  get_local_model_dict, get_global_model_dict, set_lambda_model)  # NOQA


def test_optimcfg():
    opt_cfg = OptimizerConfigurator(
        optimizer_class=SGD,
        scheduler_kwargs={
            "step_size": 1,
            "gamma": 0.1
        },
        lr=0.1,
        momentum=0.9
    )

    assert opt_cfg.optimizer == SGD
    assert opt_cfg.scheduler_kwargs == {
        "step_size": 1,
        "gamma": 0.1
    }

    assert opt_cfg.optimizer_kwargs == {
        "lr": 0.1,
        "momentum": 0.9
    }

    opt, sch = opt_cfg(Linear(10, 10))

    assert isinstance(opt, SGD)
    assert isinstance(sch, StepLR)
    assert opt.defaults["lr"] == 0.1
    assert opt.defaults["momentum"] == 0.9
    assert sch.step_size == 1
    assert sch.gamma == 0.1
    assert str(opt_cfg) == "OptCfg(SGD,lr=0.1,momentum=0.9,StepLR(step_size=1,gamma=0.1))"


def test_functions():
    try:
        client_module = import_module_from_str("fl_bench.client")
        client_class = get_class_from_str("fl_bench.client", "Client")
        model = get_model("MNIST_2NN")
        model2 = get_model("fl_bench.nets.MNIST_2NN")
        linear = get_class_from_qualified_name("torch.nn.Linear")
        full_linear = get_full_classname(Linear)
        loss = get_loss("CrossEntropyLoss")
        scheduler = get_scheduler("StepLR")
        clear_cache()
        clear_cache(True)
    except Exception:
        pytest.fail("Unexpected error!")

    assert client_module.__name__ == "fl_bench.client"
    assert client_class == Client
    assert model.__class__.__name__ == "MNIST_2NN"
    assert model2.__class__.__name__ == "MNIST_2NN"
    assert linear == Linear
    assert full_linear == "torch.nn.modules.linear.Linear"
    assert isinstance(loss, CrossEntropyLoss)
    assert scheduler == StepLR


def test_configuration():

    cfg = dict({
        'protocol': {
            'n_clients': 100,
            'n_rounds': 50,
            'eligible_perc': 0.1
        },
        'data': {
            'dataset': {
                'name': 'mnist'
            },
            'standardize': False,
            'distribution': {
                'name': "iid"
            },
            'client_split': 0.1,
            'sampling_perc': 1
        },
        'exp': {
            'seed': 42,
            'average': 'micro',
            'device': 'cpu'
        },
        'logger': {
            'name': 'local'
        }
    })
    cfg_alg = dict({
        'name': 'fl_bench.algorithms.fedavg.FedAVG',
        'hyperparameters': {
            'server': {
                'weighted': True
            },
            'client': {
                'batch_size': 10,
                'local_epochs': 5,
                'loss': 'CrossEntropyLoss',
                'optimizer': {
                    'lr': 0.01,
                    'momentum': 0.9,
                    'weight_decay': 0.0001
                },
                'scheduler': {
                    'step_size': 1,
                    'gamma': 1
                }
            },
            'model': 'MNIST_2NN'
        }
    })

    temp_cfg = tempfile.NamedTemporaryFile(mode="w")
    temp_cfg_alg = tempfile.NamedTemporaryFile(mode="w")
    json.dump(cfg, open(temp_cfg.name, "w"))
    json.dump(cfg_alg, open(temp_cfg_alg.name, "w"))

    try:
        conf = Configuration(temp_cfg.name, temp_cfg_alg.name)
    except Exception:
        pytest.fail("Unexpected error!")

    assert conf.protocol.n_clients == 100
    # assert conf.data.dataset.name == "mnist"
    assert conf.exp.seed == 42
    # assert conf.logger.name == "local"

    assert str(conf) == "fl_bench.algorithms.fedavg.FedAVG" + \
        "_data(mnist, iid)_proto(C100, R50,E0.1)_seed(42)"

    cfg = dict({"protocol": {}, "data": {}, "exp": {}, "logger": {}})
    cfg_alg = dict({"name": "fl_bench.algorithms.fedavg.FedAVG", "hyperparameters": {
                   "server": {}, "client": {}, "model": "MNIST_2NN"}})

    temp_cfg = tempfile.NamedTemporaryFile(mode="w")
    temp_cfg_alg = tempfile.NamedTemporaryFile(mode="w")
    json.dump(cfg, open(temp_cfg.name, "w"))
    json.dump(cfg_alg, open(temp_cfg_alg.name, "w"))

    with pytest.raises(ValueError):
        conf = Configuration(temp_cfg.name, temp_cfg_alg.name)


def test_log():
    log = Log()
    log.init()

    try:
        log.start_round(1, None)
        log.selected_clients(1, [1, 2, 3])
        log.message_received(Message("test", "test", None))
        log.error("test")
        log.end_round(1, {"accuracy": 1}, [{"accuracy": 0.7}, {"accuracy": 0.5}])
        log.finished([{"accuracy": 0.7}, {"accuracy": 0.5}, {"accuracy": 0.6}])
        temp = tempfile.NamedTemporaryFile(mode="w")
        log.save(temp.name)
    except Exception:
        pytest.fail("Unexpected error!")

    with open(temp.name, "r") as f:
        data = dict(json.load(f))
        assert data == {'perf_global': {'1': {'accuracy': 1}}, 'comm_costs': {
            '0': 0, '1': 4}, 'perf_local': {'1': {'accuracy': 0.6}, '2': {'accuracy': 0.6}}}

    assert log.history[1] == {"accuracy": 1}
    assert log.client_history[1] == {"accuracy": 0.6}
    assert log.comm_costs[1] == 4


def test_wandb_log():
    log2 = WandBLog()
    log2.init()
    try:
        log2.start_round(1, None)
        log2.selected_clients(1, [1, 2, 3])
        log2.message_received(Message("test", "test", None))
        log2.error("test")
        log2.end_round(1, {"accuracy": 1}, [{"accuracy": 0.7}, {"accuracy": 0.5}])
        log2.finished([{"accuracy": 0.7}, {"accuracy": 0.5}, {"accuracy": 0.6}])
        temp = tempfile.NamedTemporaryFile(mode="w")
        log2.save(temp.name)
    except Exception:
        pytest.fail("Unexpected error!")

    with open(temp.name, "r") as f:
        data = dict(json.load(f))
        assert data == {'perf_global': {'1': {'accuracy': 1}}, 'comm_costs': {
            '0': 0, '1': 4}, 'perf_local': {'1': {'accuracy': 0.6}, '2': {'accuracy': 0.6}}}

    assert log2.history[1] == {"accuracy": 1}
    assert log2.client_history[1] == {"accuracy": 0.6}
    assert log2.comm_costs[1] == 4


def test_models():
    model1 = Linear(2, 1)
    model1.weight.data.fill_(1)
    model1.bias.data.fill_(1)

    model2 = Linear(2, 1)
    model2.weight.data.fill_(2)
    model2.bias.data.fill_(2)

    model3 = merge_models(model1, model2, 0.5)
    assert model3.weight.data[0, 0] == 1.5
    assert model3.weight.data[0, 1] == 1.5
    assert model3.bias.data[0] == 1.5

    model4 = merge_models(model1, model2, 0.75)
    assert model4.weight.data[0, 0] == 1.75
    assert model4.weight.data[0, 1] == 1.75
    assert model4.bias.data[0] == 1.75

    diffdict = diff_model(model1.state_dict(), model2.state_dict())
    assert diffdict["weight"].data[0, 0] == -1.0
    assert diffdict["weight"].data[0, 1] == -1.0
    assert diffdict["bias"].data[0] == -1.0


def test_mixing():
    model1 = Linear(2, 1)
    model1.weight.data.fill_(1)
    model1.bias.data.fill_(1)

    model2 = Linear(2, 1)
    model2.weight.data.fill_(2)
    model2.bias.data.fill_(2)
    mixed = mix_networks(model1, model2, 0.5)

    x = torch.FloatTensor([[1, 1]])
    y = mixed(x)
    assert y[0, 0] == 4.5

    assert hasattr(mixed, "lam")
    assert mixed.lam == 0.5
    assert mixed.get_lambda() == 0.5
    assert diff_model(get_local_model_dict(mixed), model2.state_dict())["weight"].data[0, 0] == 0.0
    assert diff_model(get_global_model_dict(mixed), model1.state_dict())["weight"].data[0, 0] == 0.0

    set_lambda_model(mixed, 0.3)

    assert mixed.lam == 0.3
    assert mixed.get_lambda() == 0.3

    weights = mixed.get_weight()
    assert weights[0].data[0, 0] == 0.7 + 0.3 * 2
    assert weights[0].data[0, 1] == 0.7 + 0.3 * 2
    assert weights[1].data[0] == 0.7 + 0.3 * 2

    model1 = MNIST_2NN()
    model2 = MNIST_2NN()
    mixed = mix_networks(model1, model2, 0.2)

    assert mixed.get_lambda() == 0.2

    model1 = VGG9()
    model2 = VGG9()
    mixed = mix_networks(model1, model2, 0.3)
    assert mixed.get_lambda() == 0.3

    x = torch.randn(1, 1, 28, 28)
    mixed(x)

    model1 = Shakespeare_LSTM()
    model2 = Shakespeare_LSTM()
    mixed = mix_networks(model1, model2, 0.4)
    assert mixed.get_lambda() == 0.4

    x = torch.randint(0, 100, (1, 10))
    mixed(x)


if __name__ == "__main__":
    test_optimcfg()
    test_functions()
    test_configuration()
    test_log()
    # test_wandb_log()
    test_models()
    test_mixing()
