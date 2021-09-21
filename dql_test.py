import gym
import ma_gym
from dql import DQAgent

# env.step(agent_actions) -> obs_n, reward_n, done_n, info
# Full usage here https://github.com/koulanurag/ma-gym/wiki/Usage
env = gym.make("Switch2-v0")
agent1 = DQAgent(2, 5, 1)
agent2 = DQAgent(2, 5, 1)

done_n = [False, False]

while not all(done_n):
    pass
