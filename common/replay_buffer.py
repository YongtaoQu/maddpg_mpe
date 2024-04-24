import random
from collections import namedtuple, deque

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done', 'hidden_state'))


class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self._storage = deque([], maxlen=capacity)

    def add(self, state, action, reward, next_state, done, hidden_state=None):
        # hidden states is optional for rnn use.
        transition = Transition(state, action, reward, next_state, done, hidden_state)
        self._storage.append(transition)

    def sample(self, batch_size):
        transitions = random.sample(self._storage, batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, hidden_batch = zip(*transitions)
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch, hidden_batch

    def __len__(self):
        return len(self._storage)
