import os
from collections import deque

import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
import random
from environment.env import SumoEnv
from memory import Memory
from neural_net import DQN
import numpy as np

MEMORY_SIZE = 10000
BATCH_SIZE = 128
DISC_FACTOR = 0.9
UPDATE_FREQ = 1  # How often to perform a training step.

START_E = 1  # start chance of random action
END_E = 0.01  # end change of random action

ANNEALING_STEPS = 10000  # how many steps of training to reduce START_E to END_E
NUM_EPISODES = 1500  # how many episodes of game environment to train network with.
PRE_TRAIN_STEPS = 1000  # how many steps of random actions before training begins.

MAX_EP_LEN = 500  # The max allowed length of our episode
TAU = 0.006  # Rate to update target network toward primary network

LR = 0.0001


def train_all_batches():
    for i_batch, batch in enumerate(memory.batch_sampler(BATCH_SIZE)):
        with torch.no_grad():
            next_value = torch.max(target_net(batch['next_state']), dim=1)[0]

        actual_value = torch.where(
            batch['done'],
            batch['reward'],
            batch['reward'] + DISC_FACTOR * next_value).unsqueeze(-1)

        my_value = net(batch['state']).gather(1, batch['action'].unsqueeze(-1))

        optimizer.zero_grad()
        loss = loss_fn(actual_value, my_value)
        print(loss)
        loss.backward()
        optimizer.step()


def update_target_net():
    for target_param, param in zip(target_net.parameters(), net.parameters()):
        target_param.data.copy_(TAU * param.data + target_param.data * (1.0 - TAU))


with SumoEnv(render=False) as env:
    memory = Memory(MEMORY_SIZE)

    net = DQN()
    target_net = DQN()
    target_net.eval()

    eps = START_E
    total_steps = 0
    step_drop = (START_E - END_E) / ANNEALING_STEPS

    optimizer = Adam(net.parameters(), lr=LR)
    loss_fn = MSELoss()

    rewards_queue = deque(maxlen=100)

    for ep in range(NUM_EPISODES):
        state = env.reset()

        ep_len = 0
        done = False
        while not done and ep_len < MAX_EP_LEN:
            ep_len += 1
            total_steps += 1
            if random.random() < eps or total_steps < PRE_TRAIN_STEPS:
                action = env.action_space.sample()

            else:
                tensor_state = torch.tensor([state], dtype=torch.float32)
                action = net(tensor_state).max(1)[1][0].cpu().detach().numpy()


            next_state, reward, done = env.step(action)
            rewards_queue.append(reward)
            print(total_steps, np.mean(rewards_queue))

            memory.add_experience(state, action, reward, next_state, done)

            if total_steps > PRE_TRAIN_STEPS:
                if eps > END_E:
                    eps -= step_drop

                if total_steps % UPDATE_FREQ == 0:
                    train_all_batches()
                    update_target_net()
