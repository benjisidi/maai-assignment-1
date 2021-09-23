import numpy as np
import gym
import ma_gym
from dql import DQAgent
import torch
from ma_gym.wrappers import Monitor
from tqdm import trange
import shutil
# env.step(agent_actions) -> obs_n, reward_n, done_n, info
# Full usage here https://github.com/koulanurag/ma-gym/wiki/Usage
env = gym.make("Switch2-v0")
shutil.rmtree("recordings")
env = Monitor(env, directory='recordings',
              video_callable=lambda episode_id: episode_id % 100 == 0)
agent1 = DQAgent(2, 5, 1, epsilon=0.1)
agent2 = DQAgent(2, 5, 1, epsilon=0.1)

pbar = trange(5000, desc="Prev Reward: ", leave=True)
for episode in pbar:
    explore = env.episode_id % 100 == 0
    total_reward = 0
    done_n = [False, False]
    obs_1, obs_2 = env.reset()
    act_1 = agent1.act(torch.tensor(obs_1), explore=explore)
    act_2 = agent2.act(torch.tensor(obs_2), explore=explore)
    obs_n, reward_n, done_n, info = env.step([act_1, act_2])
    counter = 1
    while not all(done_n):
        act_1 = agent1.step(reward_n[0], obs_n[0])
        act_2 = agent2.step(reward_n[1], obs_n[1])
        obs_n, reward_n, done_n, info = env.step([act_1, act_2])
        counter += 1
        total_reward += np.sum(reward_n)
    pbar.set_description(f"Prev Reward: {total_reward:.0f}", refresh=True)
