import torch

from tqdm import tqdm
from copy import deepcopy
from abc import ABC, abstractmethod
from collections import OrderedDict
from tianshou.data import to_numpy, to_torch

from offlinerl.utils.env import get_env
from offlinerl.utils.net.common import MLP
from offlinerl.evaluation.neorl import test_on_real_env
from offlinerl.evaluation.d4rl import d4rl_score

class CallBackFunction:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.is_initialized = False
    
    def initialize(self, train_buffer, val_buffer, *args, **kwargs):
        self.is_initialized = True

    def __call__(self, policy) -> dict:
        assert self.is_initialized, "`initialize` should be called before calls."
        raise NotImplementedError

class PeriodicCallBack(CallBackFunction):
    '''This is a wrapper for callbacks that are only needed to perform periodically.'''
    def __init__(self, callback : CallBackFunction, period : int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._callback = callback
        self.period = period
        self.call_count = 0

    def __getattr__(self, name : str):
        return getattr(self._callback, name)

    def initialize(self, train_buffer, val_buffer, *args, **kwargs):
        self._callback.initialize(train_buffer, val_buffer, *args, **kwargs)

    def __call__(self, policy) -> dict:
        assert self._callback.is_initialized, "`initialize` should be called before calls."
        self.call_count += 1
        if self.call_count % self.period == 0:
            return self._callback(policy)
        else:
            return {}

class CallBackFunctionList(CallBackFunction):
    # TODO: run `initialize` and `__call__` in parallel
    def __init__(self, callback_list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.callback_list = callback_list

    def initialize(self, train_buffer, val_buffer, *args, **kwargs):
        for callback in self.callback_list:
            callback.initialize(train_buffer, val_buffer, *args, **kwargs)
        self.is_initialized = True

    def __call__(self, policy) -> dict:
        eval_res = OrderedDict()

        for callback in self.callback_list:
            eval_res.update(callback(policy))

        return eval_res

class OnlineCallBackFunction(CallBackFunction):
    def initialize(self, train_buffer, val_buffer, task, number_of_runs=10, seed=25, *args, **kwargs):
        self.task = task
        self.env = get_env(self.task, seed)
        self.is_initialized = True
        self.number_of_runs = number_of_runs

    def __call__(self, policy, epoch, mean, std) -> dict:
        assert self.is_initialized, "`initialize` should be called before callback."
        policy = deepcopy(policy).cpu()
        eval_res = OrderedDict()
        eval_res.update(test_on_real_env(self.task, epoch, d4rl_score, policy, mean, std, self.env, number_of_runs=self.number_of_runs))
        return eval_res

def get_defalut_callback(*args, **kwargs):
    return CallBackFunctionList([
        OnlineCallBackFunction(*args, **kwargs),
    ])