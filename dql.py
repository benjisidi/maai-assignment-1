
import random
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
        self.prev_obs = None
        self.prev_action = None
        self.epsilon = epsilon
        self.loss_fn = torch.nn.MSELoss()

    def get_loss(self, prev_state, prev_action, reward, next_state, dead):
        pred = self.model.forward(prev_state)[prev_action]
        pred_next = self.model.forward(next_state)
        updated_pred = reward + (1-dead) * self.discount * torch.max(pred_next)
        loss = self.loss_fn(pred, updated_pred)
        return loss

    def act(self, obs, explore):
        next_values = self.model.forward(obs)
        self.prev_obs = obs
        if explore:
            if np.random.rand() < self.epsilon:
                action = np.random.choice(range(self.action_space))
                self.prev_action = action
                return action
        self.prev_action = torch.argmax(next_values)
        return torch.argmax(next_values)

    def step(self, reward, next_state, dead, explore=True):
        loss = self.get_loss(self.prev_obs, self.prev_action,
                             reward, next_state, dead)
        loss.backward()
        self.optimizer.step()
        return self.act(next_state, explore=explore)


class DQAgentExperience:
    def __init__(self, state_size, action_space, discount, epsilon, batch_size):
        self.state_size = state_size
        self.action_space = action_space
        self.model = get_q_network(state_size, action_space)
        self.optimizer = torch.optim.Adam(params=self.model.parameters())
        self.discount = discount
        self.prev_obs = None
        self.prev_action = None
        self.epsilon = epsilon
        self.loss_fn = torch.nn.MSELoss()
        self.replay_buffer = []
        self.batch_size = batch_size

    def get_loss(self, prev_state, prev_action, reward, next_state, dead):
        preds = self.model.forward(prev_state)
        selected_preds = torch.gather(preds, 1, prev_action.unsqueeze(1))
        pred_next = self.model.forward(next_state)
        updated_preds = reward + (1-dead) * \
            self.discount * torch.max(pred_next)
        loss = self.loss_fn(selected_preds.squeeze(), updated_preds)
        return loss

    def act(self, obs, explore):
        next_values = self.model.forward(obs)
        self.prev_obs = obs
        if explore:
            action = torch.tensor(
                np.random.choice(range(self.action_space)))
            self.prev_action = action
            return action
        self.prev_action = torch.argmax(next_values)
        return torch.argmax(next_values)

    def step(self, reward, next_state, dead, episode, eval=False):
        if not eval:
            if callable(self.epsilon):
                explore = np.random.rand() < self.epsilon(episode)
            else:
                explore = np.random.rand() < self.epsilon
        else:
            explore = False
        self.save(reward, next_state, dead)
        return self.act(next_state, explore=explore)

    def save(self, reward, next_state, dead):
        self.replay_buffer.append(
            [self.prev_obs, self.prev_action, reward, next_state, dead])

    def learn(self):
        replays = random.choices(
            self.replay_buffer, k=self.batch_size)
        obs, acts, rewards, next_states, deads = zip(*replays)
        self.optimizer.zero_grad()
        losses = self.get_loss(torch.stack(obs), torch.stack(acts),
                               torch.tensor(rewards), torch.stack(next_states), torch.tensor(deads, dtype=int))
        losses.backward()
        self.optimizer.step()


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
