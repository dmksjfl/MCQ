# Mildly Conservative $Q$-learning (MCQ) for Offline Reinforcement Learning

Original PyTorch implementation of **MCQ** (NeurIPS 2022) from [Mildly Conservative Q-learning for Offline Reinforcement Learning](https://arxiv.org/pdf/2206.04745). The code is highly based on the [offlineRL](https://agit.ai/Polixir/OfflineRL) repository.

## Install

To use this codebase, one need to install the following dependencies:

- fire
- loguru
- tianshou==0.4.2
- gym<=0.18.3
- mujoco-py==2.0.2.8
- sklearn
- gtimer
- torch==1.8.0
- d4rl==1.1
- rlkit==0.2.1dev

Once you have all the dependencies installed, run the following command

```
pip install -e .
```

## How to run

For MuJoCo tasks, we conduct experiments on d4rl MuJoCo "-v2" datasets by calling
```
python examples/train_d4rl.py --algo_name=MCQ --task d4rl-hopper-medium-replay-v2 --seed 6 --lam 0.9 --log-dir=logs/hopper-medium-replay/r6
```

For Adroit "-v0"/maze2d "-v1" tasks, we run on these datasets by calling
```
python examples/train_d4rl.py --algo_name=MCQ --task d4rl-maze2d-medium-v1 --seed 6 --lam 0.9 --log-dir=logs/maze2d-medium-v1/r6
```

The log is stored in the `--log-dir`. One can see the training curve via tensorboard.

To modify the number of sampled actions, specify `--num` tag, default is 10. To add normalization to offline data, specify `--normalize` tag (but this is not required).

## Instruction

In the paper and our implementation, we update the critics via:
$\mathcal{L}\_{critic} = \lambda \mathbb{E}\_{s,a,s^\prime\sim\mathcal{D},a^\prime\sim\pi(\cdot|s^\prime)}[(Q(s,a) - y)^2] + (1-\lambda)\mathbb{E}\_{s\sim\mathcal{D},a\sim\pi(\cdot|s)}[(Q(s,a) - y^\prime)^2]$. While one can also try to update the critic via: $\mathcal{L}\_{critic} = \mathbb{E}\_{s,a,s^\prime\sim\mathcal{D},a^\prime\sim\pi(\cdot|s^\prime)}[(Q(s,a) - y)^2] + \alpha\mathbb{E}\_{s\sim\mathcal{D},a\sim\pi(\cdot|s)}[(Q(s,a) - y^\prime)^2]$. It is also reasonable since we ought not to let $\lambda=0$. At this time, $\alpha = \frac{1-\lambda}{\lambda}, \lambda\in(0,1)$. Note that the hyperparameter scale would vastly change using $\alpha$ (e.g., if we let $\lambda = 0.1, \alpha=9$ while if $\lambda=0.5, \alpha=1$).

We do welcome the reader to try running with $\alpha$-style update rule.

## Citation

If you use our method or code in your research, please consider citing the paper as follows:
```
@inproceedings{lyu2022mildly,
 title={Mildly Conservative Q-learning for Offline Reinforcement Learning},
 author={Jiafei Lyu and Xiaoteng Ma and Xiu Li and Zongqing Lu},
 booktitle={Thirty-sixth Conference on Neural Information Processing Systems},
 year={2022}
}
```
