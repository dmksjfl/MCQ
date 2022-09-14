# Mildly Conservative $Q$-learning (MCQ) for Offline Reinforcement Learning

Original PyTorch implementation of **MCQ**. The code is highly based on the [offlineRL](https://agit.ai/Polixir/OfflineRL) repository.

## Install

To use this codebase, one need to install the following dependencies:

- fire
- loguru
- tianshou
- gym<=0.18.3
- mujoco-py
- sklearn
- gtimer
- torch==1.8.0
- d4rl
- rlkit

Once you have all the dependencies installed, run the following command

```
pip install -e .
```

## How to run

For MuJoCo tasks, we conduct experiments on d4rl MuJoCo "-v2" datasets, run
```
python examples/train_d4rl.py --algo_name=MCQ --task d4rl-hopper-medium-replay-v2 --seed 6 --lam 0.9 --log-dir=logs/hopper-medium-replay/r6
```

For Adroit/maze2d tasks, we run on  "-v0" datasets, run
```
python examples/train_d4rl.py --algo_name=MCQ --task d4rl-pen-human-v0 --seed 6 --lam 0.3 --log-dir=logs/pen-human/r6
```

The log is stored in the `--log-dir`. One can see the training curve via tensorboard.

To modify the number of sampled actions, specify `--num` tag, default is 10. To add normalization to offline data, specify `--normalize` tag (but this is not required).
