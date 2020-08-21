from models.neural_net import DQN, SimpleLinear
from models.frap import Frap

registered_models = [DQN, SimpleLinear, Frap]
model_types_names = {
    model.__name__: model for model in registered_models
}


def get_save_dict(model):
    return {
        'model_class_name': model.__class__.__name__,
        'model_save_dict': model.get_save_dict()
    }


def load_model_from_dict(model_dict):
    # print(model_dict)
    # print(model_types_names)
    return model_types_names[model_dict['model_class_name']].load_from_dict(model_dict['model_save_dict'])
