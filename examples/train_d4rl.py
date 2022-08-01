import fire

from offlinerl.algo import algo_select
from offlinerl.data.d4rl import load_d4rl_buffer
from offlinerl.evaluation import OnlineCallBackFunction
from offlinerl.utils.log import Logger

import torch
import numpy as np
import random


def run_algo(**kwargs):
    algo_init_fn, algo_trainer_obj, algo_config = algo_select(kwargs)
    train_buffer = load_d4rl_buffer(algo_config["task"])

    tblogger = Logger(algo_config["log_dir"], use_tb=True)

    ## setup seed
    torch.manual_seed(algo_config["seed"])
    np.random.seed(algo_config["seed"])
    torch.cuda.manual_seed_all(algo_config["seed"])
    random.seed(algo_config["seed"])

    algo_init = algo_init_fn(algo_config)
    algo_trainer = algo_trainer_obj(algo_init, algo_config)
    callback = OnlineCallBackFunction()
    callback.initialize(train_buffer=train_buffer, val_buffer=None, task=algo_config["task"], seed=algo_config["seed"])
    algo_trainer.train(train_buffer, None, callback_fn=callback, tblogger=tblogger)

    tblogger._sw.close()

if __name__ == "__main__":
    fire.Fire(run_algo)
    