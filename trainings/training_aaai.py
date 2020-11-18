from pathlib import Path
from time import sleep

from torch.nn import MSELoss
from torch.optim import Adam

from environments.sumo_env import SumoEnv
from evaluation.evaluator import Evaluator
from memory.prioritized_memory import Memory
from models.frap import Frap
from models.neural_net import SimpleLinear
from settings import JSONS_FOLDER
from trainings.training import get_model_name, main_train
from trainings.training_parameters import TrainingParameters, TrainingState


def get_new_training():
    traffic_movements = 12
    traffic_phases = 8

    model_name = get_model_name('simple')
    training_param = TrainingParameters(
        model_name,
        pre_train_steps=1500,
        tau=0.001,
        lr=0.0001
    )

    memory = Memory(training_param.memory_size)

    net = SimpleLinear(traffic_movements + 1, traffic_phases)
    target_net = SimpleLinear(traffic_movements + 1, traffic_phases)
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
        memory,
        ['gneJ18']
    )


def get_frap_training():
    model_name = get_model_name('frap')
    training_param = TrainingParameters(
        model_name,
        pre_train_steps=1500,
        tau=0.001,
        lr=0.0001,
        save_freq=1,
        test_freq=50
    )

    memory = Memory(training_param.memory_size)

    net = Frap(32, 16, 16, 2, 16)
    target_net = Frap(32, 16, 16, 2, 16)

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
        memory,
        ['gneJ18']
    )


def train_aaai():
    env_config_path = Path(JSONS_FOLDER, 'configs', 'aaai_random.json')
    evaluator = Evaluator.from_file(Path(JSONS_FOLDER, 'evaluators', 'example_test.json'))
    main_train(
        get_frap_training(),
        SumoEnv.from_config_file(env_config_path),
        evaluator,
        Path('saved', 'aaai-random', 'frap'),
    )


def train_2v2():
    env_config_path = Path(JSONS_FOLDER, 'configs', 'all_equal.json')

    training_state = get_frap_training()
    training_state.junctions = ['gneJ25', 'gneJ26', 'gneJ27', 'gneJ28']

    main_train(
        training_state,
        SumoEnv.from_config_file(env_config_path),
        None,
        Path('saved', 'aaai', 'frap'),
    )

if __name__ == '__main__':
    #train_aaai()
    train_2v2()
