import math
import random
import torch
import numpy as np

from memory.PER import PER


class Memory:
    def get_save_dict(self):
        return {
            'init_params': {'buffer_size': self.buffer_size, 'state_shape': self.state_shape},
            'buffer': self.buffer
        }

    @classmethod
    def load_from_dict(cls, dict_to_load):
        m = Memory(**dict_to_load['init_params'])
        m.buffer = dict_to_load['buffer']
        return m

    def __init__(self, buffer_size, state_shape=(40, 40, 2)):
        self.buffer_size = buffer_size
        self.state_shape = state_shape
        self.buffer = PER(self.buffer_size)

    def update(self, idx, error):
        self.buffer.update(idx, error)

    # add normalization in future
    def prepare_state(self, state):
        if not isinstance(state, torch.Tensor):
            return torch.tensor(state, dtype=torch.float32)
        return state

    def get_empty_state(self):
        return torch.zeros(self.state_shape, dtype=torch.float32)

    def add_experience(self, training_state, state, action, reward, next_state, done, device):

        params = training_state.training_parameters
        model = training_state.model
        target_model = training_state.target_model

        value = model(torch.tensor([state], dtype=torch.float32, device=device))[0][action]

        target_qvals = target_model(torch.tensor([next_state], dtype=torch.float32, device=device))

        target_value = reward
        if not done:
            target_value += params.disc_factor * torch.max(target_qvals)

        error = abs(value - target_value).cpu().detach().numpy()

        state = np.array(state)  # self.prepare_state(state)
        next_state = np.array(next_state)  # self.prepare_state(next_state)
        action = np.array(action)  # torch.tensor([action], dtype=torch.long)
        reward = np.array(reward)  # torch.tensor([reward], dtype=torch.float32)
        done = np.array(done)  # torch.tensor([done], dtype=torch.bool)

        item = [state, action, reward, next_state, done]
        self.buffer.add(error, item)

    def __len__(self):
        return self.buffer.get_size()

    def batch_sampler(self, batch_size, device=torch.device('cpu')):
        return PrioritizedMemBatchSampler(self.buffer, batch_size, device)


class PrioritizedMemBatchSampler:
    def __init__(self, buffer: PER, batch_size, device):
        self.buffer = buffer
        self.batch_size = batch_size
        self.device = device

        self.i_batch = 0
        self.num_batches = 0

        self.order = ['id', 'state', 'action', 'reward', 'next_state', 'done', 'is_weight']
        self.d_types = [torch.short, torch.float32, torch.long, torch.float32, torch.float32, torch.bool, torch.float32]

    def __iter__(self):
        self.i_batch = 0
        self.num_batches = math.ceil(self.buffer.get_size() / self.batch_size)
        return self

    def __next__(self):
        if self.i_batch < self.num_batches:
            self.i_batch += 1
            output_dict = {}
            my_batch = np.array(self.buffer.sample(self.batch_size), dtype=object)
            for i, (name, d_type) in enumerate(zip(self.order, self.d_types)):
                my_vals = np.stack(my_batch[:, i])
                output_dict[name] = torch.tensor(my_vals, dtype=d_type, device=self.device)
            return output_dict
        raise StopIteration
