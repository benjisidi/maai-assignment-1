
import torch
import numpy as np


def get_q_network(state_action_dims, action_space, hidden_size=100):
    return torch.nn.Sequential(
        torch.nn.Linear(state_action_dims, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(
            int(hidden_size),
            int(hidden_size/2)
        ),
        torch.nn.ReLU(),
        torch.nn.Linear(int(hidden_size/2),
                        action_space),
    )


class DQAgent:
    def __init__(self, state_size, action_space, discount, epsilon):
        self.state_size = state_size
        self.action_space = action_space
        self.model = get_q_network(state_size, action_space)
        self.optimizer = torch.optim.Adam(params=self.model.parameters())
        self.discount = discount
        self.prev_output = None
        self.epsilon = epsilon

    def pseudo_loss(self, output, reward, next_state):
        return 0.5 * (reward + self.discount * (torch.max(self.model.forward(torch.tensor(next_state)))) - output)

    def act(self, obs, explore):
        next_values = self.model.forward(obs)
        if explore:
            if np.random.rand() < self.epsilon:
                action = np.random.choice(range(self.action_space))
                self.prev_output = next_values[action]
                return action
        self.prev_output = torch.max(next_values)
        return torch.argmax(next_values)

    def step(self, reward, next_state, explore=True):
        loss = self.pseudo_loss(self.prev_output, reward, next_state)
        loss.backward()
        self.optimizer.step()
        return self.act(torch.tensor(next_state), explore=explore)


class DQAgentExperience:
    def __init__(self, state_size, action_space, discount, epsilon):
        self.state_size = state_size
        self.action_space = action_space
        self.model = get_q_network(state_size, action_space)
        self.optimizer = torch.optim.Adam(params=self.model.parameters())
        self.discount = discount
        self.prev_output = None
        self.epsilon = epsilon
        self.replay_buffer = []

    def pseudo_loss(self, output, reward, next_state):
        return 0.5 * (reward + self.discount * (torch.max(self.model.forward(torch.tensor(next_state)))) - output)

    def act(self, obs, explore):
        next_values = self.model.forward(obs)
        if explore:
            if np.random.rand() < self.epsilon:
                action = np.random.choice(range(self.action_space))
                self.prev_output = next_values[action]
                return action
        self.prev_output = torch.max(next_values)
        return torch.argmax(next_values)

    def step(self, reward, next_state, explore=True):
        loss = self.pseudo_loss(self.prev_output, reward, next_state)
        loss.backward()
        self.optimizer.step()
        return self.act(torch.tensor(next_state), explore=explore)


if __name__ == "__main__":
    model = get_q_network(100, 2)
    optimiser = torch.optim.Adam(params=model.parameters())
    loss = torch.nn.HuberLoss()
    state_action_pair = torch.rand(100)
    scores = model.forward(state_action_pair)
    output = loss(scores, torch.tensor([1, 0]).float())
    output.backward()
    optimiser.step()
    print(model)
