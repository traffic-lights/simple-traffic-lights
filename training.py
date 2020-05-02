import os

import torch

from environment.env import SumoEnv
from memory import Memory
from neural_net import DQN

memory = Memory(10000)

net = DQN()

with SumoEnv(render=False) as env:
    state = env.reset()
    state = torch.tensor([state], dtype=torch.float32)
    print(net(state))
