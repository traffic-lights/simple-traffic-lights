from pathlib import Path

import torch

from environment.aaai_env import AaaiEnv
from environment.simple_env import SimpleEnv
from trainings.training_parameters import TrainingState

state = TrainingState.from_path(
    Path('saved', 'aaai', 'frap', 'frap_2020-10-02.19-08-44-396561', 'states',
         'ep_6_frap_2020-10-02.19-08-44-396561.tar'))

model = state.model

with AaaiEnv(render=True, save_replay=False) as env:
    state = env.reset()
    ep_len = 0
    done = False

    prev_action = 0

    while not done:
        ep_len += 1
        tensor_state = torch.tensor([state], dtype=torch.float32)
        action = model(tensor_state).max(1)[1][0].cpu().detach().numpy().item()

        next_state, reward, done, info = env.step(action)

        state = next_state
        if prev_action == action:
            print("takie samo", action)
        else:
            print("rozne", action)
        prev_action = action
