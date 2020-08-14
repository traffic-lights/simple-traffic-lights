from pathlib import Path

import torch

from environment.env import SumoEnv
from environment.simple_env import SimpleEnv
from training_parameters import TrainingState

state = TrainingState.from_path(
    Path('saved', 'ddqn_2020-07-18.14-21-48-424555', 'states', 'ep_30_ddqn_2020-07-18.14-21-48-424555.tar'))

model = state.model

with SimpleEnv(render=True, save_replay=True) as env:
    state = env.reset()
    ep_len = 0
    done = False
    while not done:
        ep_len += 1
        tensor_state = torch.tensor([state], dtype=torch.float32)
        action = model(tensor_state).max(1)[1][0].cpu().detach().numpy()

        next_state, reward, done, info = env.step(action)

        state = next_state

        print(env.get_throughput())
