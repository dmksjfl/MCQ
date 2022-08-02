import os
import uuid

from offlinerl.utils.io import create_dir

def log_path():
    import offlinerl
    log_path = os.path.abspath(os.path.join(offlinerl.__file__,"../../","offlinerl_tmp"))

    create_dir(log_path)

    return log_path
    
