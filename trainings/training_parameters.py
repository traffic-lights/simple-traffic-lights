from dataclasses import dataclass, asdict

import torch
from torch import nn
from torch.optim import Optimizer, Adam

from memory import Memory
from models.utils import get_save_dict, load_model_from_dict
from models.neural_net import SerializableModel


@dataclass
class TrainingParameters:
    model_name: str

    memory_size: int = 10000
    batch_size: int = 128
    disc_factor: float = 0.9
    training_freq: int = 1  # How often to perform a training step.

    start_e: float = 1  # start chance of random action
    end_e: float = 0.01  # end change of random action

    annealing_steps: int = 10000  # how many steps of training to reduce start_e to end_e
    num_episodes: int = 1500  # how many episodes of game environment to train network with.
    pre_train_steps: int = 1000  # how many steps of random actions before training begins.

    max_ep_len: int = 300  # The max allowed length of our episode

    tau: float = 0.0003  # Rate to update target network toward primary network
    target_update_freq: int = 1  # how often to preform a target net update

    lr: float = 0.0001

    total_steps: int = 0  # how many steps performed
    current_eps: float = start_e
    current_episode: int = 0
    save_freq: int = 10  # how often save (current_episode % save_freq == 0)

    def __post_init__(self):
        self.step_drop = (self.start_e - self.end_e) / self.annealing_steps


def get_optimizer_dict(optimizer):
    return {
        'optim_class_name': optimizer.__class__.__name__,
        'optim_save_dict': optimizer.state_dict()
    }


optim_class_mapper = {
    'Adam': Adam
}


def load_optim_from_dict(dict_to_load, model: nn.Module):
    # print(dict_to_load)
    my_otpim = optim_class_mapper[dict_to_load['optim_class_name']](model.parameters())
    my_otpim.load_state_dict(dict_to_load['optim_save_dict'])
    return my_otpim


@dataclass
class TrainingState:
    training_parameters: TrainingParameters
    model: SerializableModel
    target_model: SerializableModel
    optimizer: Optimizer
    loss_fn: nn.Module
    replay_memory: Memory

    def save(self, path):
        my_dict = {
            'model': get_save_dict(self.model),
            'target_model': get_save_dict(self.target_model),
            'optimizer': get_optimizer_dict(self.optimizer),
            'loss_fn': self.loss_fn,
            'replay_memory': self.replay_memory.get_save_dict(),
            'training_parameters': asdict(self.training_parameters)
        }
        torch.save(my_dict, path)

    @classmethod
    def from_path(cls, path):
        my_dict = torch.load(path, map_location='cpu')
        model = load_model_from_dict(my_dict['model'])
        target_model = load_model_from_dict(my_dict['target_model'])

        optim = load_optim_from_dict(my_dict['optimizer'], model)
        mem = Memory.load_from_dict(my_dict['replay_memory'])

        training_parameters = TrainingParameters(**my_dict['training_parameters'])

        return TrainingState(
            training_parameters,
            model,
            target_model,
            optim,
            my_dict['loss_fn'],
            mem
        )
