import random
import numpy as np
from memory.SumTree import SumTree


class PER:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6
    start_beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.beta = PER.start_beta
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def get_size(self):
        return self.tree.n_entries

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / (n + 1)
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()

        result = []
        for idx, sample, is_weight in zip(idxs, batch, is_weights):
            result.append([idx] + sample + [is_weight])

        return result

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)