from torch import nn

from models.neural_net import DQN


class SerializableModel(nn.Module):
    def get_save_dict(self):
        raise NotImplementedError

    @classmethod
    def load_from_dict(cls, dict_to_load):
        raise NotImplementedError


def get_save_dict(model):
    return {
        'model_class_name': model.__class__.__name__,
        'model_save_dict': model.get_save_dict()
    }


def load_model_from_dict(model_dict):
    return model_types_names[model_dict['model_class_name']].load_from_dict(model_dict['model_save_dict'])


model_types_names = {
    'DQN': DQN
}