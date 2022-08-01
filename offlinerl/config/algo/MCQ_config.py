import torch
from offlinerl.utils.exp import select_free_cuda

task = "d4rl-hopper-medium-replay-v2"
seed = 25

device = 'cuda' if torch.cuda.is_available() else 'cpu'
obs_shape = None
act_shape = None
max_action = None

max_epoch = 1000
steps_per_epoch = 1000

vae_features = 750
vae_layers = 2
vae_lr = 1e-3
vae_step = 200000

## new add
log_dir = './logs/MCQ/halfcheetah-medium-v2'

batch_size = 256
hidden_layer_size = 400
layer_num = 2
actor_lr=3E-4
critic_lr=3E-4
reward_scale=1
use_automatic_entropy_tuning=True
target_entropy = None
discount = 0.99
soft_target_tau=5e-3

lam = 0.7
num = 10