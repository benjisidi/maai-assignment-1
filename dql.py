
import random
import torch
import numpy as np


def get_q_network(state_action_dims, action_space, hidden_size=64):
    return torch.nn.Sequential(
        torch.nn.Linear(state_action_dims, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(
            int(hidden_size),
            int(hidden_size*2)
        ),
        torch.nn.ReLU(),
        torch.nn.Dropout(),
        torch.nn.Linear(
            int(hidden_size*2),
            int(hidden_size*4)
        ),
        torch.nn.ReLU(),
        torch.nn.Dropout(),
        torch.nn.Linear(int(hidden_size*4),
                        action_space),
    )


class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = []
        self.weights = np.array([])
        self.max_size = max_size

    def save(self, memory, weight):
        self.buffer.append(memory)
        self.weights.append()
        self.weights.append()
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def sample(self, k):
        return random.sample(self.buffer, k=k)

    def __len__(self):
        return len(self.buffer)


class NumpyReplayBuffer:
    def __init__(self, max_len, width):
        self.buffer = np.empty((max_len, width))
        self.weights = np.zeros(max_len)
        self.insertion_index = 0
        self.max_len = max_len
        self.full = False

    def save(self, row, weight):
        self.buffer[self.insertion_index] = row
        self.weights[self.insertion_index] = weight

        if self.insertion_index == self.max_len - 1:
            self.insertion_index = 0
            self.full = True
        else:
            self.insertion_index += 1

    def sample(self, k):
        sample_idx = np.random.choice(
            np.arange(self.max_len, dtype=int), size=k, replace=False, p=self.weights/np.sum(self.weights))
        examples = self.buffer[sample_idx]
        return examples, sample_idx

    def update_weights(self, new_weights, idx):
        self.weights[idx] = new_weights

    def __len__(self):
        return self.insertion_index if not self.full else self.max_len


class DQN:
    def __init__(self, state_size, action_space, discount, epsilon, batch_size, lr, update_steps, memory_size, max_noop):
        self.state_size = state_size
        self.action_space = action_space
        self.model = get_q_network(state_size, action_space)
        self.target_model = get_q_network(state_size, action_space)
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=lr)
        self.discount = discount
        self.prev_obs = None
        self.prev_action = None
        self.epsilon = epsilon
        self.loss_fn = torch.nn.MSELoss()
        # S+S' = state_size *2, action + reward + done = 3
        self.replay_buffer = NumpyReplayBuffer(memory_size, state_size * 2 + 3)
        self.batch_size = batch_size
        self.update_steps = update_steps
        self.update_counter = 0
        self.max_noop = max_noop
        self.noop_count = 0

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_loss(self, prev_state, prev_action, reward, next_state, dead):
        preds = self.model.forward(prev_state)
        selected_preds = torch.gather(
            preds, 1, prev_action.unsqueeze(1).to(torch.int64))
        pred_next = self.target_model.forward(next_state).detach()
        target = reward + (1-dead) * \
            self.discount * torch.max(pred_next)
        delta = selected_preds.squeeze() - target
        loss = self.loss_fn(selected_preds.squeeze(), target)
        return loss, np.absolute(delta.detach().numpy())

    def act(self, obs, episode):
        if callable(self.epsilon):
            explore = np.random.rand() <= self.epsilon(episode)
        else:
            explore = np.random.rand() <= self.epsilon
        next_values = self.model.forward(obs)
        action = torch.argmax(next_values)
        if action == 4:
            if self.noop_count >= self.max_noop:
                explore = True
                self.noop_count = 0
            else:
                self.noop_count += 1

        else:
            self.noop_count = 0
        if explore:
            action = torch.tensor(
                np.random.choice(range(self.action_space)))
        self.prev_action = action
        self.prev_obs = obs
        return action

    def step(self, reward, next_state, dead, episode):
        self.save(reward, next_state, dead)
        return self.act(next_state, episode)

    def save(self, reward, next_state, dead):
        # New experiences are saved with a very large weight to ensure each is seen at least once
        self.replay_buffer.save(
            [*self.prev_obs.numpy(), self.prev_action.item(), reward, *next_state.numpy(), int(dead)], 99999)

    def learn(self):
        if self.update_counter % self.update_steps == 0:
            self.update_target_model()
        replays, idx = self.replay_buffer.sample(k=self.batch_size)
        replays = torch.tensor(replays).float()
        obs = replays[:, :self.state_size]
        acts = replays[:, self.state_size]
        rewards = replays[:, self.state_size + 1]
        next_states = replays[:, self.state_size + 2: self.state_size*2 + 2]
        deads = replays[:, -1].int()
        self.optimizer.zero_grad()
        losses, new_weights = self.get_loss(
            obs, acts, rewards, next_states, deads)
        self.replay_buffer.update_weights(new_weights, idx)
        losses.backward()
        self.optimizer.step()
        self.update_counter += 1


class DDQN:
    def __init__(self, state_size, action_space, discount, epsilon, batch_size, lr, update_steps, memory_size, max_noop):
        self.state_size = state_size
        self.action_space = action_space
        self.model_A = get_q_network(state_size, action_space)
        self.model_B = get_q_network(state_size, action_space)
        self.optimizer_A = torch.optim.Adam(
            params=self.model_A.parameters(), lr=lr)
        self.optimizer_B = torch.optim.Adam(
            params=self.model_B.parameters(), lr=lr)
        self.discount = discount
        self.prev_obs = None
        self.prev_action = None
        self.epsilon = epsilon
        self.loss_fn = torch.nn.MSELoss()
        self.replay_buffer = ReplayBuffer(memory_size)
        self.batch_size = batch_size
        self.update_steps = update_steps
        self.update_counter = 0
        self.max_noop = max_noop
        self.noop_count = 0

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_loss(self, prev_state, prev_action, reward, next_state, dead, pred_model, target_model):
        preds = pred_model.forward(prev_state)
        selected_preds = torch.gather(preds, 1, prev_action.unsqueeze(1))
        pred_next = target_model.forward(next_state).detach()
        target = reward + (1-dead) * \
            self.discount * torch.max(pred_next)
        loss = self.loss_fn(selected_preds.squeeze(), target)
        return loss

    def act(self, obs, episode):
        models = [self.model_A, self.model_B]
        pred_index = np.random.choice([0, 1])
        model = models[pred_index]
        if callable(self.epsilon):
            explore = np.random.rand() <= self.epsilon(episode)
        else:
            explore = np.random.rand() <= self.epsilon
        next_values = model.forward(obs)
        action = torch.argmax(next_values)
        if action == 4:
            if self.noop_count >= self.max_noop:
                explore = True
                self.noop_count = 0
            else:
                self.noop_count += 1
        else:
            self.noop_count = 0

        if explore:
            action = torch.tensor(
                np.random.choice(range(self.action_space)))

        self.prev_action = action
        self.prev_obs = obs
        return action

    def step(self, reward, next_state, dead, episode):
        self.save(reward, next_state, dead)
        return self.act(next_state, episode)

    def save(self, reward, next_state, dead):
        self.replay_buffer.save(
            [*self.prev_obs, self.prev_action, reward, *next_state, dead])

    def learn(self):
        models = [self.model_A, self.model_B]
        optimisers = [self.optimizer_A, self.optimizer_B]
        pred_index = np.random.choice([0, 1])
        target_index = int(not pred_index)
        replays, idx = self.replay_buffer.sample(k=self.batch_size)

        obs, acts, rewards, next_states, deads = zip(*replays)
        optimisers[pred_index].zero_grad()
        losses = self.get_loss(torch.stack(obs), torch.stack(acts),
                               torch.tensor(rewards), torch.stack(next_states), torch.tensor(deads, dtype=int), models[pred_index], models[target_index])
        losses.backward()
        optimisers[pred_index].step()
        self.update_counter += 1


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
