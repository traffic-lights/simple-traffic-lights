import json
from collections import defaultdict
from pathlib import Path

from environments.sumo_env import SumoEnv
from settings import PROJECT_ROOT
import numpy as np


def evaluate_controller(runner, controllers, state_mean_std=None):
    states = runner.reset()
    ep_len = 0
    done = False
    all_rewards = []
    all_states = []
    if not isinstance(controllers, list):
        controllers = [controllers] * len(runner.junctions)

    while not done:
        actions = []
        for i, controller in enumerate(controllers):
            state = states[i]
            if state_mean_std is not None:
                state = (np.array(state) - state_mean_std[0]) / state_mean_std[1]
            actions.append(controller(state))

        states, rewards, done, info = runner.step(actions)
        all_rewards.extend(info['reward'])
        # all_states.extend(states)
        ep_len += 1

    return {
        'throughput': runner.get_throughput(),
        'travel_time': runner.get_travel_time(),
        'mean_reward': np.mean(all_rewards),
        # 'std_reward': np.std(all_rewards),
        # 'mean_state': np.mean(all_states, axis=0),
        # 'std_state': np.std(all_states, axis=0)
    }


class Evaluator:
    @staticmethod
    def from_file(file_path):
        with open(file_path) as f:
            return Evaluator(json.load(f)['test_cases'])

    def __init__(self, test_cases_list):
        self.test_cases_list = test_cases_list

        self.environments = {
            test_case['name']: SumoEnv.from_config_file(Path(PROJECT_ROOT, test_case['config']),max_steps=5000)
            for test_case in self.test_cases_list
        }

    def evaluate_traffic_controllers(self, traffic_controllers, state_mean_std=None):
        """
        :param traffic_controllers: list of controllers
        :return: list of metrics for each controller, in the same order as given list of controllers
        """

        if not isinstance(traffic_controllers, list):
            traffic_controllers = [traffic_controllers]

        metrics = [{} for _ in traffic_controllers]

        for env_name, env in self.environments.items():
            with env.create_runner(render=True) as runner:
                for i_controller, controller in enumerate(traffic_controllers):
                    if not isinstance(controller, list):
                        controller = [controller]
                    controller = [c.with_connection(runner.connection) for c in controller]

                    metrics[i_controller][env_name] = evaluate_controller(runner, controller, state_mean_std)

        return metrics

    def evaluate_to_tensorboard(self, traffic_controllers_dict, tf_writer, step, state_mean_std=None):
        controller_names = list(traffic_controllers_dict.keys())
        controllers = list(traffic_controllers_dict.values())

        metrics = self.evaluate_traffic_controllers(controllers, state_mean_std)

        scalars = defaultdict(dict)

        for controller_name, metric in zip(controller_names, metrics):
            for env_name, evaluated in metric.items():
                for k, v in evaluated.items():
                    scalars[env_name + '/' + k][controller_name] = v

        for tag_name, scalar in scalars.items():
            tf_writer.add_scalars(tag_name, scalar, step)

        tf_writer.flush()
