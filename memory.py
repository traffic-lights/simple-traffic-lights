from collections import deque

import torch


class Memory:
    def __init__(self, buffer_size, state_shape=(40, 40, 2)):
        self.buffer_size = buffer_size
        self.state_shape = state_shape
        self.buffer = deque(maxlen=self.buffer_size)

    # add normalization in future
    def prepare_state(self, state):
        return state

    def get_empty_state(self):
        return torch.zeros(self.state_shape, dtype=torch.float32)

    def add_experience(self, state, action, reward, next_state, done):
        state = self.prepare_state(state)
        next_state = self.prepare_state(next_state)

        self.buffer.append({
            'state': state,
            'action': action,
            'reward': torch.tensor(reward, dtype=torch.float32),
            'next_state': next_state,
            'done': torch.tensor(done, dtype=torch.bool)
        })

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer[idx]
