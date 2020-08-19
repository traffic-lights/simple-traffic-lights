from pathlib import Path
from time import sleep

from torch.nn import MSELoss
from torch.optim import Adam

from environment.aaai_env import AaaiEnv
from memory import Memory
from models.neural_net import SimpleLinear
from trainings.training import get_model_name, main_train
from trainings.training_parameters import TrainingParameters, TrainingState


def get_new_training():
    traffic_phases = 4

    model_name = get_model_name('simple')
    training_param = TrainingParameters(model_name)

    memory = Memory(training_param.memory_size)

    net = SimpleLinear(traffic_phases * 2 + 1, traffic_phases)
    target_net = SimpleLinear(traffic_phases * 2 + 1, traffic_phases)
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


if __name__ == '__main__':
    main_train(get_new_training(), AaaiEnv, Path('saved', 'aaai', 'simple'))
