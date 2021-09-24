import numpy as np
import gym
import ma_gym
from dql import DQN, DDQN
import torch
from ma_gym.wrappers import Monitor
from tqdm import trange
import shutil
import matplotlib.pyplot as plt
from time import sleep
# env.step(agent_actions) -> obs_n, reward_n, done_n, info
# Full usage here https://github.com/koulanurag/ma-gym/wiki/Usage


def moving_average(data, window):
    return np.convolve(data, np.ones(window), 'valid') / window


# def test_naive_dqn():
#     env = gym.make("Switch2-v0")
#     shutil.rmtree("recordings")
#     env = Monitor(env, directory='recordings',
#                   video_callable=lambda episode_id: episode_id % 100 == 0)
#     agent1 = DQAgent(2, 5, 1, epsilon=0.1)
#     agent2 = DQAgent(2, 5, 1, epsilon=0.1)

#     pbar = trange(5000, desc="Prev Reward: ", leave=True)
#     for episode in pbar:
#         explore = env.episode_id % 100 == 0
#         total_reward = 0
#         done_n = [False, False]
#         obs_1, obs_2 = env.reset()
#         act_1 = agent1.act(torch.tensor(obs_1), explore=explore)
#         act_2 = agent2.act(torch.tensor(obs_2), explore=explore)
#         obs_n, reward_n, done_n, info = env.step([act_1, act_2])
#         counter = 1
#         while not all(done_n):
#             act_1 = agent1.step(reward_n[0], torch.tensor(obs_n[0]), done_n[0])
#             act_2 = agent2.step(reward_n[1], torch.tensor(obs_n[1]), done_n[1])
#             obs_n, reward_n, done_n, info = env.step([act_1, act_2])
#             counter += 1
#             total_reward += np.sum(reward_n)
#         pbar.set_description(f"Prev Reward: {total_reward:.0f}", refresh=True)


def test_replay_dqn(lr):
    batch_size = 32
    n_episodes = 40000
    epsilon_start = 1
    epsilon_end = 0.2
    decay_end = 30000
    # lr = 1e-3
    discount_factor = 0.99
    update_steps = 50
    memory_size = 15000
    update_frequency = 2
    replay_start_size = 100
    max_noop = 3
    env = gym.make("Switch2-v0")

    state_size = len(env.reset()[0])
    action_space_size = env.action_space[0].n

    def epsilon(x): return (epsilon_start-epsilon_end) * \
        ((decay_end - x)/decay_end) + \
        epsilon_end if x < decay_end else epsilon_end

    agent_class = DQN
    agents = [agent_class(state_size, action_space_size, discount=discount_factor, epsilon=epsilon,
                          batch_size=batch_size, lr=lr, update_steps=update_steps, memory_size=memory_size, max_noop=max_noop) for x in range(env.n_agents)]
    shutil.rmtree("recordings", ignore_errors=True)
    env = Monitor(env, directory='recordings',
                  video_callable=lambda episode_id: episode_id % 200 == 0)

    pbar = trange(n_episodes, desc="Rwd: E:", leave=True)
    episode_rewards = []
    for episode in pbar:
        total_reward = 0
        done_n = [False, False]
        obs_n = env.reset()
        acts = [agents[i].act(torch.tensor(obs_n[i]), episode=episode)
                for i in range(env.n_agents)]
        obs_n, reward_n, done_n, info = env.step(acts)
        counter = 1
        while not all(done_n):
            # env.render()
            acts = [agents[i].step(reward_n[i], torch.tensor(
                obs_n[i]), done_n[i], episode=episode) for i in range(env.n_agents)]
            obs_n, reward_n, done_n, info = env.step(acts)
            counter += 1
            total_reward += np.sum(reward_n)

            if len(agents[0].replay_buffer) > replay_start_size and counter % update_frequency == 0:
                [agent.learn() for agent in agents]

        for i in range(env.n_agents):
            agents[i].save(reward_n[i], torch.tensor(obs_n[i]),  done_n[i])
        episode_rewards.append(total_reward)
        pbar.set_description(
            f"Rwd: {np.mean(episode_rewards[-20:]):.2f} E: {epsilon(episode):.3f}", refresh=True)

    np.save(f"./episode_rewards_{lr:.0e}.npy", episode_rewards)
    plot_results(episode_rewards, epsilon)
    plt.title(f"Switch2-v0 DQN avg reward, lr={lr:.0e}")
    plt.savefig(f"./rewards_{lr:.0e}.png")
    # plt.show()


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
    lr = 1e-3
    test_replay_dqn(lr)
