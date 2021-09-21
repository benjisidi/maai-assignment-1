import torch


def get_q_network(state_action_dims, action_space):
    return torch.nn.Sequential(
        torch.nn.Linear(state_action_dims, int(state_action_dims/2)),
        torch.nn.ReLU(),
        torch.nn.Linear(
            int(state_action_dims/2),
            int(state_action_dims/4)
        ),
        torch.nn.ReLU(),
        torch.nn.Linear(int(state_action_dims/4),
                        action_space),
    )


class DQAgent:
    def __init__(self, state_size, action_space, discount):
        self.state_size = state_size
        self.action_space = action_space
        self.model = get_q_network(state_size, action_space)
        self.optimizer = torch.optim.Adam(params=self.model.parameters())
        self.discount = discount
        self.prev_output = 0

    def pseudo_loss(self, output, reward, next_state):
        return 0.5 * (reward + self.discount * (torch.max(self.model.forward(next_state))) - output)

    def step(self, reward, next_state):
        loss = self.pseudo_loss(self.prev_output, reward, next_state)
        loss.backward()
        self.optimizer.step()
        next_values = self.model.forward(next_state)
        self.prev_output = torch.max(next_values)
        return torch.argmax(next_values)


if __name__ == "__main__":
    model = get_q_network(100, 2)
    optimiser = torch.optim.Adam(params=model.parameters())
    loss = torch.nn.HuberLoss()
    state_action_pair = torch.rand(100)
    scores = model.forward(state_action_pair)
    output = loss(scores, torch.tensor([1, 0]).float())
    print(output)
    output.backward()
    optimiser.step()
    print(model)
