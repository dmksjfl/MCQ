import os
import uuid
import json
from abc import ABC, abstractmethod

import torch
from collections import OrderedDict
from loguru import logger
from offlinerl.utils.exp import init_exp_logger
from offlinerl.utils.io import create_dir, download_helper, read_json


class BaseAlgo(ABC):
    def __init__(self, args):        
        logger.info('Init AlgoTrainer')
        if "exp_name" not in args.keys():
            exp_name = str(uuid.uuid1()).replace("-","")
        else:
            exp_name = args["exp_name"]
        
    
    def log_res(self, epoch, result):
        logger.info('Epoch : {}', epoch)
        for k,v in result.items():
            logger.info('{} : {}',k, v)
            
    
    @abstractmethod
    def train(self, 
              history_buffer,
              eval_fn=None,):
        pass
    
    def _sync_weight(self, net_target, net, soft_target_tau = 5e-3):
        for o, n in zip(net_target.parameters(), net.parameters()):
            o.data.copy_(o.data * (1.0 - soft_target_tau) + n.data * soft_target_tau)
    
    @abstractmethod
    def get_policy(self,):
        pass
    
    def save_model(self, model_path):
        torch.save(self.get_policy(), model_path)
        
    def load_model(self, model_path):
        model = torch.load(model_path)
        
        return model