import numpy as np


class GeneralQ(object):

    def __init__(self, number_of_states, number_of_actions, initial_state,
                 target_policy, behaviour_policy, double, step_size=0.1):
        # Settings.
        self._number_of_actions = number_of_actions
        self._step_size = step_size
        self._behaviour_policy = behaviour_policy
        self._target_policy = target_policy
        self._double = double
        # Initial state.
        self._s = initial_state
        # Tabular q-estimates.
        self._q = np.zeros((number_of_states, number_of_actions))
        if double:
            self._q2 = np.zeros((number_of_states, number_of_actions))
        # The first action in an agent's lifetime is always 0(=up) in our setup.
        self._last_action = 0

    @property
    def q_values(self):
        return (self._q + self._q2)/2 if self._double else self._q

    def step(self, reward, discount, next_state):
        if self._double:
            # Choose whether we are updating q or q2
            # Use the same behaviour policy, opposite target policy
            pick = int(np.random.rand() > 0.5)
            qs = [self._q, self._q2]
        else:
            pick = 0
            qs = [self._q, self._q]
        next_action = self._behaviour_policy(qs[pick][next_state])
        target_action_probs = self._target_policy(
            qs[(pick + 1) % 2][next_state], next_action)
        target_action_val = np.sum(qs[(pick + 1) % 2] * target_action_probs)
        if pick:
            self._q2[self._s, self._last_action] += self._step_size * (
                reward + discount * target_action_val -
                self._q2[self._s, self._last_action]
            )
        else:
            self._q[self._s, self._last_action] += self._step_size * (
                reward + discount * target_action_val -
                self._q[self._s, self._last_action]
            )
        self._last_action = next_action
        self._s = next_state
        return next_action
