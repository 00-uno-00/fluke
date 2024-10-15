from fluke.data.datasets import Datasets
from fluke import DDict

dataset = Datasets.get("emnist", path="./data")

from fluke.data import DataSplitter

data = DDict( dataset=dataset,
              distribution="iid",
              sampling_perc=1,
              client_split=0.2,
              keep_test=True,
              server_test=True,
              server_split=0.0,
              uniform_test=True)

splitter = DataSplitter(**data)

from fluke.evaluation import ClassificationEval
from fluke import GlobalSettings

evaluator = ClassificationEval(eval_every=1, n_classes=dataset.num_classes)
GlobalSettings().set_evaluator(evaluator)
GlobalSettings().set_device("cuda")
GlobalSettings().set_seed(42)

client_hp = DDict(
    batch_size=20,
    local_epochs=5,
    loss="CrossEntropyLoss",
    optimizer=DDict(
      lr=0.1),
    scheduler=DDict(
      gamma=1,
      step_size=1)
)

alg_hp = DDict(
    client = client_hp,
    model="Scaffold_2FC",
    server=DDict(weighted=True))

from fluke.algorithms.scaffold import SCAFFOLD

algorithm = SCAFFOLD(100, splitter, alg_hp)

from fluke.utils.log import *

'''config = DDict(data = data,
               exp=DDict(seed=42, device="cuda"),
               eval=DDict(task="classification",
                          eval_every=1,
                          pre_fit=False,
                          post_fit=False,
                          server=True,
                          locals=False),
               logger=DDict(name="WandBLog",
                            project="fluke_benchmarks"),
               protocol=DDict(eligible_perc=0.2,
                              n_clients=100,
                              n_rounds=40),
               method=DDict(hyperparameters=alg_hp),
               name="fluke.algorithms.scaffold.SCAFFOLD"
               )
Logger = get_logger("WandBLog", **config)
'''

Logger = WandBLog(project="fluke_benchmarks")
Logger.init()
Logger.pretty_log(alg_hp, "fluke_benchmarks")

algorithm.set_callbacks(Logger)

algorithm.run(40, 0.2)
