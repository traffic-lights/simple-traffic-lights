from models.neural_net import DQN, SimpleLinear, Frap


def get_save_dict(model):
    return {
        'model_class_name': model.__class__.__name__,
        'model_save_dict': model.get_save_dict()
    }


def load_model_from_dict(model_dict):
    return model_types_names[model_dict['model_class_name']].load_from_dict(model_dict['model_save_dict'])


registered_models = [DQN, SimpleLinear, Frap]

model_types_names = {
    model.__class__.__name__: model for model in registered_models
}