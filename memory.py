import math
import random
from collections import deque

import torch
from torch.utils.data import IterableDataset
import numpy as np


class Memory:
    def __init__(self, buffer_size, state_shape=(40, 40, 2)):
        self.buffer_size = buffer_size
        self.state_shape = state_shape
        self.buffer = deque(maxlen=self.buffer_size)

    # add normalization in future
    def prepare_state(self, state):
        if not isinstance(state, torch.Tensor):
            return torch.tensor(state, dtype=torch.float32)
        return state

    def get_empty_state(self):
        return torch.zeros(self.state_shape, dtype=torch.float32)

    def add_experience(self, state, action, reward, next_state, done):
        state = np.array(state)  # self.prepare_state(state)
        next_state = np.array(next_state)  # self.prepare_state(next_state)
        action = np.array(action)  # torch.tensor([action], dtype=torch.long)
        reward = np.array(reward)  # torch.tensor([reward], dtype=torch.float32)
        done = np.array(done)  # torch.tensor([done], dtype=torch.bool)

        item = np.array([state, action, reward, next_state, done])
        self.buffer.append(item)

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer[idx]

    def batch_sampler(self, batch_size, device=torch.device('cpu')):
        return MemBatchSampler(self.buffer, batch_size, device)


class MemBatchSampler:
    def __init__(self, dequeue, batch_size, device):
        self.dequeue = dequeue
        self.batch_size = batch_size
        self.device = device

        self.i_batch = 0
        self.num_batches = 0

        self.order = ['state', 'action', 'reward', 'next_state', 'done']
        self.d_types = [torch.float32, torch.long, torch.float32, torch.float32, torch.bool]

    def __iter__(self):
        self.i_batch = 0
        self.num_batches = math.ceil(len(self.dequeue) / self.batch_size)
        return self

    def __next__(self):
        if self.i_batch < self.num_batches:
            self.i_batch += 1
            output_dict = {}
            my_batch = np.array(random.sample(self.dequeue, self.batch_size))
            for i, (name, d_type) in enumerate(zip(self.order, self.d_types)):
                my_vals = np.stack(my_batch[:, i])
                output_dict[name] = torch.tensor(my_vals, dtype=d_type, device=self.device)
            return output_dict
        raise StopIteration
