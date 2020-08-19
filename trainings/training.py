import random
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn import MSELoss
from torch.optim import Adam

from environment.simple_env import SimpleEnv
from memory import Memory
from models.neural_net import DQN
from torch.utils.tensorboard import SummaryWriter

from trainings.training_parameters import TrainingParameters, TrainingState


def train_all_batches(memory, net, target_net, optimizer, loss_fn,
                      batch_size, disc_factor, device=torch.device('cpu')):
    net = net.train()
    losses = []
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
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

    return np.mean(losses)


def update_target_net(net, target_net, tau):
    for target_param, param in zip(target_net.parameters(), net.parameters()):
        target_param.data.copy_(tau * param.data + target_param.data * (1.0 - tau))


def test_model(net, env, max_ep_len, device, should_random=False):
    net = net.eval()
    rewards = []

    with torch.no_grad():

        state = env.reset()
        ep_len = 0
        done = False

        while not done and ep_len < max_ep_len:
            if should_random:
                action = env.action_space.sample()
            else:
                tensor_state = torch.tensor([state], dtype=torch.float32, device=device)
                action = net(tensor_state).max(1)[1].cpu().detach().numpy()[0]
            state, reward, done, info = env.step(action)
            rewards.append(reward)
            ep_len += 1

        return {
            'throughput': env.get_throughput(),
            'travel_time': env.get_travel_time(),
            'mean_reward': np.mean(rewards)
        }


def get_model_name(suffix):
    return suffix + '_' + datetime.now().strftime("%Y-%m-%d.%H-%M-%S-%f")


def load_training_path(path):
    pass


def get_new_training():
    model_name = get_model_name('ddqn')
    training_param = TrainingParameters(model_name)

    memory = Memory(training_param.memory_size)

    net = DQN()
    target_net = DQN()
    target_net.eval()

    target_net.load_state_dict(net.state_dict())
    optimizer = Adam(net.parameters(), lr=training_param.lr)

    loss_fn = MSELoss()
    return TrainingState(
        training_param,
        net,
        target_net,
        optimizer,
        loss_fn,
        memory
    )


def main_train(training_state: TrainingState, env_class=SimpleEnv, save_root=Path('saved', 'old_models')):
    params = training_state.training_parameters

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print("training on: {}".format(device))

    training_state.model.to(device)
    training_state.target_model.to(device)

    rewards_queue = deque(maxlen=300)

    save_root = Path(save_root, params.model_name)

    state_save_root = Path(save_root, 'states')
    state_save_root.mkdir(exist_ok=True, parents=True)

    tensorboard_save_root = Path(save_root, 'tensorboard')
    tensorboard_save_root.mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter(tensorboard_save_root)
    with env_class(render=False) as env:

        while params.current_episode < params.num_episodes:

            state = env.reset()

            ep_len = 0
            done = False
            while not done and ep_len < params.max_ep_len:
                ep_len += 1
                params.total_steps += 1
                if random.random() < params.current_eps or params.total_steps < params.pre_train_steps:
                    action = env.action_space.sample()

                else:
                    tensor_state = torch.tensor([state], dtype=torch.float32, device=device)
                    action = training_state.model(tensor_state).max(1)[1][0].cpu().detach().numpy()

                next_state, reward, done, info = env.step(action)

                rewards_queue.append(reward)
                print(params.total_steps, np.mean(rewards_queue))

                training_state.replay_memory.add_experience(state, action, reward, next_state, done)
                state = next_state

                if params.total_steps > params.pre_train_steps:
                    if params.current_eps > params.end_e:
                        params.current_eps -= params.step_drop

                    if params.total_steps % params.training_freq == 0:
                        mean_loss = train_all_batches(
                            training_state.replay_memory,
                            training_state.model, training_state.target_model,
                            training_state.optimizer,
                            training_state.loss_fn, params.batch_size, params.disc_factor, device=device)

                        writer.add_scalar('Train/Loss', mean_loss, params.total_steps)

                    if params.total_steps % params.target_update_freq == 0:
                        update_target_net(training_state.model, training_state.target_model, params.tau)

            if params.total_steps < params.pre_train_steps:
                random_test = True
            else:
                random_test = False

            test_stats = test_model(training_state.model, env, params.max_ep_len, device,
                                    random_test)

            training_state.model = training_state.model.train()

            for k, v in test_stats.items():
                if random_test:
                    name = 'Test/random/{}'.format(k)
                else:
                    name = 'Test/model/{}'.format(k)
                writer.add_scalar(name, v, params.current_episode)

            writer.flush()

            params.current_episode += 1

            if params.current_episode % params.save_freq == 0:
                training_state.save(
                    Path(state_save_root, 'ep_{}_{}.tar'.format(params.current_episode, params.model_name))
                )

        training_state.save(
            Path(state_save_root, 'final_{}.tar'.format(params.current_episode, params.model_name))
        )


if __name__ == '__main__':
    main_train(get_new_training())
