import os
import uuid
import random

import torch
import numpy as np
from loguru import logger

from offlinerl.utils.logger import log_path

def setup_seed(seed=1024):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
        
def select_free_cuda():
    tmp_name = str(uuid.uuid1()).replace("-","")
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >'+tmp_name)
    memory_gpu = [int(x.split()[2]) for x in open(tmp_name, 'r').readlines()]
    os.system('rm '+tmp_name) 
    
    return np.argmax(memory_gpu)

def set_free_device_fn():
    device = 'cuda'+":"+str(select_free_cuda()) if torch.cuda.is_available() else 'cpu'

    return device


def init_exp_logger(repo=None, experiment_name=None, flush_frequency=1):
    pass