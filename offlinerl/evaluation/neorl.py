import multiprocessing
import gym
import ray
from copy import deepcopy
import numpy as np
from collections import OrderedDict

from offlinerl.utils.env import get_env

def test_one_trail(env, policy, mean, std, epoch, number_of_runs=10):

    lengths = []
    avg_reward = 0.
    _num_of_runs = number_of_runs
    for _ in range(_num_of_runs):
        state, done = env.reset(), False
        state = (state - mean)/std
        step = 0
        while not done:
            state = state[np.newaxis]
            action = policy.get_action(state).reshape(-1)
            state, reward, done, _ = env.step(action)
            avg_reward += reward
            step += 1
        lengths.append(step)
    ave_length = np.mean(lengths)
    avg_reward /= number_of_runs
    d4rl_score = env.get_normalized_score(avg_reward) * 100
    
    return (avg_reward, d4rl_score, ave_length)

def test_on_real_env(task, epoch, scorer, policy, mean, std, env, number_of_runs=10):
    rewards, d4rl_score, episode_lengths = test_one_trail(env, policy, mean, std, epoch, number_of_runs)

    res = OrderedDict()
    res["Reward_Mean_Env"] = rewards
    res["Length_Mean_Env"] = episode_lengths
    res["score"] = d4rl_score

    return res
