import random
from collections import deque

import numpy as np
import torch
from torch.nn import MSELoss
from torch.optim import Adam

from environment.env import SumoEnv
from memory import Memory
from neural_net import DQN


def train_all_batches(memory, net, target_net, optimizer, loss_fn, batch_size, disc_factor, device=torch.device('cpu')):
    for i_batch, batch in enumerate(memory.batch_sampler(batch_size, device=device)):
        with torch.no_grad():
            next_value = torch.max(target_net(batch['next_state']), dim=1)[0]

        actual_value = torch.where(
            batch['done'],
            batch['reward'],
            batch['reward'] + disc_factor * next_value).unsqueeze(-1)

        my_value = net(batch['state']).gather(1, batch['action'].unsqueeze(-1))

        optimizer.zero_grad()
        loss = loss_fn(actual_value, my_value)
        print(loss)
        loss.backward()
        optimizer.step()


def update_target_net(net, target_net, tau):
    for target_param, param in zip(target_net.parameters(), net.parameters()):
        target_param.data.copy_(tau * param.data + target_param.data * (1.0 - tau))


def main_train():
    memory_size = 10000
    batch_size = 128
    disc_factor = 0.9
    update_freq = 1  # How often to perform a training step.

    start_e = 1  # start chance of random action
    end_e = 0.01  # end change of random action

    annealing_steps = 10000  # how many steps of training to reduce start_e to end_e
    num_episodes = 1500  # how many episodes of game environment to train network with.
    pre_train_steps = 1000  # how many steps of random actions before training begins.

    max_ep_len = 500  # The max allowed length of our episode
    tau = 0.006  # Rate to update target network toward primary network

    lr = 0.0001

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print("training on: {}".format(device))

    memory = Memory(memory_size)

    net = DQN()
    target_net = DQN()
    target_net.eval()

    target_net.load_state_dict(net.state_dict())

    net.to(device)
    target_net.to(device)

    eps = start_e
    total_steps = 0
    step_drop = (start_e - end_e) / annealing_steps

    optimizer = Adam(net.parameters(), lr=lr)
    loss_fn = MSELoss()

    rewards_queue = deque(maxlen=100)

    with SumoEnv(render=False) as env:

        for ep in range(num_episodes):
            state = env.reset()


            ep_len = 0
            done = False
            while not done and ep_len < max_ep_len:
                ep_len += 1
                total_steps += 1
                if random.random() < eps or total_steps < pre_train_steps:
                    action = env.action_space.sample()

                else:
                    tensor_state = torch.tensor([state], dtype=torch.float32, device=device)
                    action = net(tensor_state).max(1)[1][0].cpu().detach().numpy()

                next_state, reward, done, info = env.step(action)

                rewards_queue.append(reward)
                print(total_steps, np.mean(rewards_queue))

                memory.add_experience(state, action, reward, next_state, done)

                if total_steps > pre_train_steps:
                    if eps > end_e:
                        eps -= step_drop

                    if total_steps % update_freq == 0:
                        train_all_batches(memory, net, target_net, optimizer,
                                          loss_fn, batch_size, disc_factor, device=device)
                        update_target_net(net, target_net, tau)


if __name__ == '__main__':
    main_train()
