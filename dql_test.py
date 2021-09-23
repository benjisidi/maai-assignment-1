import numpy as np
import gym
import ma_gym
from dql import DQAgent, DQAgentExperience
import torch
from ma_gym.wrappers import Monitor
from tqdm import trange
import shutil
import matplotlib.pyplot as plt
# env.step(agent_actions) -> obs_n, reward_n, done_n, info
# Full usage here https://github.com/koulanurag/ma-gym/wiki/Usage


def test_naive_dqn():
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
            act_1 = agent1.step(reward_n[0], torch.tensor(obs_n[0]), done_n[0])
            act_2 = agent2.step(reward_n[1], torch.tensor(obs_n[1]), done_n[1])
            obs_n, reward_n, done_n, info = env.step([act_1, act_2])
            counter += 1
            total_reward += np.sum(reward_n)
        pbar.set_description(f"Prev Reward: {total_reward:.0f}", refresh=True)


def test_replay_dqn():
    batch_size = 32
    n_episodes = 20000
    epsilon_start = 0.2
    epsilon_end = 0.05
    decay_end = 12000

    env = gym.make("Switch2-v0")
    shutil.rmtree("recordings", ignore_errors=True)
    shutil.rmtree("final_recordings", ignore_errors=True)
    env = Monitor(env, directory='recordings',
                  video_callable=lambda episode_id: episode_id % 200 == 0)
    # Up the frequency of recordings for the final few episodes
    env = Monitor(env, directory='final_recordings',
                  video_callable=lambda episode_id: episode_id > n_episodes-100 and episode_id % 10 == 0)

    def epsilon(x): return (epsilon_start-epsilon_end) * \
        (x/decay_end) + epsilon_end if x < decay_end else epsilon_end
    agent1 = DQAgentExperience(2, 5, 1, epsilon=epsilon, batch_size=batch_size)
    agent2 = DQAgentExperience(2, 5, 1, epsilon=epsilon, batch_size=batch_size)

    pbar = trange(n_episodes, desc="Prev Reward: ", leave=True)
    episode_rewards = []
    eval_ = False
    for episode in pbar:
        total_reward = 0
        done_n = [False, False]
        obs_1, obs_2 = env.reset()
        act_1 = agent1.act(torch.tensor(obs_1), episode=episode, eval=eval_)
        act_2 = agent2.act(torch.tensor(obs_2), episode=episode, eval=eval_)
        obs_n, reward_n, done_n, info = env.step([act_1, act_2])
        counter = 1
        while not all(done_n):
            act_1 = agent1.step(reward_n[0], torch.tensor(
                obs_n[0]), done_n[0], episode=episode, eval=eval_)
            act_2 = agent2.step(reward_n[1], torch.tensor(
                obs_n[1]), done_n[1], episode=episode, eval=eval_)
            obs_n, reward_n, done_n, info = env.step([act_1, act_2])
            counter += 1
            total_reward += np.sum(reward_n)
        agent1.save(reward_n[0], torch.tensor(obs_n[0]),  done_n[0])
        agent2.save(reward_n[1], torch.tensor(obs_n[1]),  done_n[1])
        episode_rewards.append(total_reward)
        pbar.set_description(
            f"Avg Reward: {np.mean(episode_rewards[-20:]):2.0f}", refresh=True)

        if len(agent1.replay_buffer) > batch_size:
            agent1.learn()
            agent2.learn()

    plt.plot(range(len(episode_rewards)), episode_rewards)
    plt.show()


if __name__ == "__main__":
    test_replay_dqn()
