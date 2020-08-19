from pathlib import Path

import torch

from environment.aaai_env import AaaiEnv
from environment.simple_env import SimpleEnv
from trainings.training_parameters import TrainingState

state = TrainingState.from_path(
    Path('saved', 'aaai', 'simple', 'simple_2020-08-19.13-08-55-996950', 'states',
         'ep_40_simple_2020-08-19.13-08-55-996950.tar'))

model = state.model

with AaaiEnv(render=True, save_replay=True) as env:
    state = env.reset()
    ep_len = 0
    done = False
    while not done:
        ep_len += 1
        tensor_state = torch.tensor([state], dtype=torch.float32)
        action = model(tensor_state).max(1)[1][0].cpu().detach().numpy().item()

        next_state, reward, done, info = env.step(action)

        state = next_state

        print(env.get_throughput())
