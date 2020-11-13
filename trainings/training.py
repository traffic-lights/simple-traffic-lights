import random
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn import MSELoss
from torch.optim import Adam

from environments.simple_env import SimpleEnv
from environments.sumo_env import SumoEnv
from evaluation.evaluator import Evaluator
from memory.prioritized_memory import Memory
from models.frap import Frap
from models.neural_net import DQN
from torch.utils.tensorboard import SummaryWriter

from traffic_controllers.model_controller import ModelController
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

        errors = torch.abs(actual_value - my_value).cpu().detach().numpy()

        # update priority
        for i, idx in enumerate(batch['id']):
            memory.update(idx, errors[i])

        optimizer.zero_grad()
        loss = (batch['is_weight'] * loss_fn(actual_value, my_value)).mean()
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

    return np.mean(losses)


def update_target_net(net, target_net, tau):
    for target_param, param in zip(target_net.parameters(), net.parameters()):
        target_param.data.copy_(tau * param.data + target_param.data * (1.0 - tau))


def get_model_name(suffix):
    return suffix + '_' + datetime.now().strftime("%Y-%m-%d.%H-%M-%S-%f")


def main_train(training_state: TrainingState, env: SumoEnv, evaluator: Evaluator = None,
               save_root=Path('saved', 'old_models')):
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
    print(tensorboard_save_root.resolve())
    writer = SummaryWriter(tensorboard_save_root)
    with env.create_runner(render=False) as runner:

        while params.current_episode < params.num_episodes:

            state = runner.reset()

            ep_len = 0
            done = False
            while not done and ep_len < params.max_ep_len:
                ep_len += 1
                params.total_steps += 1
                if random.random() < params.current_eps or params.total_steps < params.pre_train_steps:
                    action = runner.action_space.sample()

                else:
                    tensor_state = torch.tensor([state], dtype=torch.float32, device=device)
                    action = training_state.model(tensor_state).max(1)[1].cpu().detach().numpy()[0].item()

                next_state, reward, done, info = runner.step(action)
                rewards_queue.append(reward)
                print(params.total_steps, np.mean(rewards_queue))

                training_state.replay_memory.add_experience(training_state, state, action, reward, next_state, done,
                                                            device)
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

                    if evaluator is not None and params.total_steps % params.test_freq == 0:
                        evaluator.evaluate_to_tensorboard(
                            {'model': ModelController(training_state.model.eval(), device)},
                            writer,
                            params.total_steps
                        )
                        training_state.model = training_state.model.train()

            writer.flush()

            params.current_episode += 1

            if params.current_episode % params.save_freq == 0:
                training_state.save(
                    Path(state_save_root, 'ep_{}_{}.tar'.format(params.current_episode, params.model_name))
                )

        training_state.save(
            Path(state_save_root, 'final_{}.tar'.format(params.current_episode, params.model_name))
        )
