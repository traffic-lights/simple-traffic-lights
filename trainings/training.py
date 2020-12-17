import random
from collections import deque
from datetime import datetime
from pathlib import Path
from contextlib import ExitStack

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR, StepLR

from evaluation.evaluator import Evaluator
from torch.utils.tensorboard import SummaryWriter

from traffic_controllers.model_controller import ModelController
from trainings.training_parameters import TrainingParameters, TrainingState


def train_all_batches(memory, beta,
                      prioritized_replay_eps, net,
                      target_net, optimizer, loss_fn,
                      batch_size, disc_factor, device=torch.device('cpu'), num_batches=1):
    experience = memory.sample(batch_size, beta)
    states, actions, rewards, next_states, dones, weights, batch_idxs = experience

    states = torch.tensor(states, dtype=torch.float32, device=device)
    actions = torch.tensor(actions, dtype=torch.long, device=device)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
    dones = torch.tensor(dones, dtype=torch.bool, device=device)
    weights = torch.tensor(weights, dtype=torch.float32, device=device)

    net = net.train()

    # for i_batch, batch in enumerate(memory.batch_sampler(batch_size, device=device)):
    with torch.no_grad():
        next_value = torch.max(target_net(next_states), dim=1)[0]

    actual_value = torch.where(
        dones,
        rewards,
        rewards + disc_factor * next_value).unsqueeze(-1)

    my_value = net(states).gather(1, actions.unsqueeze(-1))

    errors = torch.abs(actual_value - my_value) \
        .cpu() \
        .detach() \
        .numpy()

    new_priorities = np.abs(errors) + prioritized_replay_eps
    memory.update_priorities(batch_idxs, new_priorities)

    optimizer.zero_grad()
    loss = loss_fn(actual_value, my_value).squeeze()
    loss = (weights * loss).mean()

    loss.backward()
    # torch.nn.utils.clip_grad.clip_grad_norm_(net.parameters(), 10)
    # torch.nn.utils.clip_grad_norm_(net.parameters(), 2.)
    optimizer.step()
    return loss.cpu().item()


def update_target_net(net, target_net, tau):
    for target_param, param in zip(target_net.parameters(), net.parameters()):
        target_param.data.copy_(tau * param.data + target_param.data * (1.0 - tau))


def get_model_name(suffix):
    return suffix + '_' + datetime.now().strftime("%Y-%m-%d.%H-%M-%S-%f")


def normalize_states(states, state_mean_std):
    if state_mean_std is None:
        return states
    return [
        (np.array(s) - state_mean_std[0]) / state_mean_std[1]
        for s in states
    ]


def main_train(training_state: TrainingState, envs, evaluator: Evaluator = None,
               save_root=Path('saved', 'old_models')):
    params = training_state.training_parameters

    if not isinstance(envs, list):
        envs = [envs]

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

    # per_step_lr_drop = 0.9 / 150000
    # scheduler = LambdaLR(training_state.optimizer, lambda step: max(0.1, 1. - step * per_step_lr_drop))
    scheduler = StepLR(training_state.optimizer, step_size=250000, gamma=0.2)

    with ExitStack() as stack:
        runners = [stack.enter_context(env.create_runner(render=False)) for env in envs]
        runner_it = 0

        while params.current_episode < params.num_episodes:

            runner = runners[runner_it]
            runner_it = (runner_it + 1) % len(envs)

            not_none_prev_states = runner.reset()
            prev_states = not_none_prev_states.copy()

            not_none_prev_states = normalize_states(not_none_prev_states, training_state.state_mean_std)
            prev_states = normalize_states(prev_states, training_state.state_mean_std)

            prev_actions = [None] * len(training_state.junctions)

            ep_len = 0
            done = False
            while not done:  # and ep_len < params.max_ep_len:
                ep_len += 1
                params.total_steps += 1

                if random.random() < params.current_eps or params.total_steps < params.pre_train_steps:
                    actions = runner.action_space.sample().tolist()
                    for i, state in enumerate(prev_states):
                        if state is None:
                            actions[i] = None
                else:
                    actions = []
                    for state in prev_states:
                        if state is None:
                            actions.append(None)
                        else:
                            tensor_state = torch.tensor([state], dtype=torch.float32, device=device)
                            actions.extend(training_state.model(tensor_state).max(1)[1].cpu().detach().numpy().tolist())

                next_states, rewards, done, info = runner.step(actions)
                next_states = normalize_states(next_states, training_state.state_mean_std)
                rewards_queue.extend(info['reward'])
                print(params.total_steps, np.mean(rewards_queue))

                for s, r, a, n_s in zip(not_none_prev_states, info['reward'], prev_actions, next_states):
                    if n_s is not None:
                        if training_state.reward_mean_std is not None:
                            r = (r - training_state.reward_mean_std[0]) / training_state.reward_mean_std[1]

                        training_state.replay_memory.add(
                            s, a, r, n_s, done
                        )

                prev_states = next_states

                for i, next_state in enumerate(next_states):
                    if next_state is not None:
                        not_none_prev_states[i] = next_state

                for i, action in enumerate(actions):
                    if action is not None:
                        prev_actions[i] = action

                if params.total_steps > params.pre_train_steps:
                    if params.current_eps > params.end_e:
                        params.current_eps -= params.step_drop

                    params.sampler_current_beta += params.beta_inc
                    params.sampler_current_beta = min(params.sampler_current_beta, params.sampler_beta_max)

                    if params.total_steps % params.training_freq == 0:
                        mean_loss = train_all_batches(
                            training_state.replay_memory,
                            params.sampler_current_beta,
                            params.prioritized_replay_eps,
                            training_state.model, training_state.target_model,
                            training_state.optimizer,
                            training_state.loss_fn, params.batch_size, params.disc_factor, device=device)
                        writer.add_scalar('Train/Loss', mean_loss, params.total_steps)

                    if params.total_steps % params.target_update_freq == 0:
                        update_target_net(training_state.model, training_state.target_model, params.tau)

                    if evaluator is not None and params.total_steps % params.test_freq == 0:
                        # if params.total_steps < 15000 and params.total_steps % (params.test_freq*2) != 0:
                        #     continue
                        evaluator.evaluate_to_tensorboard(
                            {'model': ModelController(training_state.model.eval(), device)},
                            writer,
                            params.total_steps,
                            training_state.state_mean_std
                        )
                        training_state.model = training_state.model.train()

                    scheduler.step()

                    if params.total_steps % params.save_freq == 0:
                        training_state.save(
                            Path(state_save_root, 'ep_{}_{}.tar'.format(params.total_steps, params.model_name))
                        )

            writer.flush()

            params.current_episode += 1

        training_state.save(
            Path(state_save_root, 'final_{}.tar'.format(params.model_name))
        )
