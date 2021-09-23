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


def moving_average(data, window):
    return np.convolve(data, np.ones(window), 'valid') / window


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
    n_episodes = 12000
    epsilon_start = 0.2
    epsilon_end = 0.05
    decay_end = 12000

    env = gym.make("Switch2-v0")
    eval_env = gym.make("Switch2-v0")
    shutil.rmtree("recordings", ignore_errors=True)
    shutil.rmtree("eval_recordings", ignore_errors=True)
    env = Monitor(env, directory='recordings',
                  video_callable=lambda episode_id: episode_id % 200 == 0)
    eval_env = Monitor(eval_env, directory="eval_recordings")
    def epsilon(x): return (epsilon_start-epsilon_end) * \
        ((decay_end - x)/decay_end) + epsilon_end if x < decay_end else 0.01
    agent1 = DQAgentExperience(2, 5, 1, epsilon=0.17, batch_size=batch_size)
    agent2 = DQAgentExperience(2, 5, 1, epsilon=0.17, batch_size=batch_size)

    pbar = trange(n_episodes, desc="Prev Reward: ", leave=True)
    episode_rewards = []
    eval_rewards = []
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

        if episode % 100 == 0:
            eval_reward = 0
            eval_done_n = [False, False]
            eval_obs_1, eval_obs_2 = eval_env.reset()
            eval_act_1 = agent1.act(torch.tensor(
                eval_obs_1), episode=episode, eval=True)
            eval_act_2 = agent2.act(torch.tensor(
                eval_obs_2), episode=episode, eval=True)
            eval_obs_n, eval_reward_n, eval_done_n, info = eval_env.step(
                [eval_act_1, eval_act_2])
            counter = 1
            while not all(eval_done_n):
                eval_act_1 = agent1.step(eval_reward_n[0], torch.tensor(
                    eval_obs_n[0]), eval_done_n[0], episode=episode, eval=True)
                eval_act_2 = agent2.step(eval_reward_n[1], torch.tensor(
                    eval_obs_n[1]), eval_done_n[1], episode=episode, eval=True)
                eval_obs_n, eval_reward_n, eval_done_n, info = eval_env.step([
                    eval_act_1, eval_act_2])
                counter += 1
                eval_reward += np.sum(eval_reward_n)
            eval_rewards.append(eval_reward)

    plot_results(episode_rewards, epsilon)
    plt.figure()
    plt.plot(np.arange(len(eval_rewards))*100,
             eval_rewards, label="Eval. Reward")
    plt.legend()
    plt.show()


def plot_results(rewards, epsilon, n=50):
    fig, ax = plt.subplots()

    twin1 = ax.twinx()
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg Reward")
    twin1.set_ylabel("Epsilon")

    avg_reward = moving_average(np.array(rewards), 50)
    p1, = ax.plot(range(len(avg_reward)), avg_reward, "b-",
                  label=f"Avg Reward (n={n})")
    p2, = twin1.plot(range(len(avg_reward)), [epsilon(
        x) for x in range(len(avg_reward))], "r-", label="Epsilon")

    ax.legend(handles=[p1, p2])


if __name__ == "__main__":
    test_replay_dqn()
