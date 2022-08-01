from loguru import logger
import warnings

warnings.filterwarnings('ignore')


from offlinerl.config.algo import MCQ_config
from offlinerl.utils.config import parse_config
from offlinerl.algo.modelfree import MCQ

algo_dict = {
    'MCQ' : {"algo" : MCQ, "config" : MCQ_config},
}

def algo_select(command_args, algo_config_module=None):
    algo_name = command_args["algo_name"]
    logger.info('Use {} algorithm!', algo_name)
    assert algo_name in algo_dict.keys()
    algo = algo_dict[algo_name]["algo"]
    
    if algo_config_module is None:
        algo_config_module = algo_dict[algo_name]["config"]
    algo_config = parse_config(algo_config_module)
    algo_config.update(command_args)
    
    algo_init = algo.algo_init
    algo_trainer = algo.AlgoTrainer
    
    return algo_init, algo_trainer, algo_config
    
    