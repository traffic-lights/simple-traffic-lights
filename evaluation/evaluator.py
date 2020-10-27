import json
from collections import defaultdict
from pathlib import Path

from environments.sumo_env import SumoEnv
from settings import PROJECT_ROOT
import numpy as np


def evaluate_controller(runner, controller, max_ep_len):
    state = runner.reset()
    ep_len = 0
    done = False
    rewards = []

    while not done and ep_len < max_ep_len:
        action = controller(state)

        state, reward, done, info = runner.step(action)
        rewards.append(reward)
        ep_len += 1

    return {
        'throughput': runner.get_throughput(),
        'travel_time': runner.get_travel_time(),
        'mean_reward': np.mean(rewards)
    }


class Evaluator:
    @staticmethod
    def from_file(file_path):
        with open(file_path) as f:
            return Evaluator(json.load(f)['test_cases'])

    def __init__(self, test_cases_list):
        self.test_cases_list = test_cases_list

        self.environments = {
            test_case['name']: SumoEnv.from_config_file(Path(PROJECT_ROOT, test_case['config']))
            for test_case in self.test_cases_list
        }

    def evaluate_traffic_controllers(self, traffic_controllers):
        """
        :param traffic_controllers: list of controllers
        :return: list of metrics for each controller, in the same order as given list of controllers
        """

        if not isinstance(traffic_controllers, list):
            traffic_controllers = [traffic_controllers]

        metrics = [{} for _ in traffic_controllers]

        for env_name, env in self.environments.items():
            with env.create_runner(render=False) as runner:
                for i_controller, controller in enumerate(traffic_controllers):
                    metrics[i_controller][env_name] = evaluate_controller(runner, controller, max_ep_len=300)

        return metrics

    def evaluate_to_tensorboard(self, traffic_controllers_dict, tf_writer, step):
        controller_names = traffic_controllers_dict.keys()
        controllers = traffic_controllers_dict.valies()

        metrics = self.evaluate_traffic_controllers(controllers)

        scalars = defaultdict(dict)

        for controller_name, metric in zip(controller_names, metrics):
            for env_name, evaluated in metric.items():
                for k, v in evaluated.items():
                    scalars[env_name + '/' + k][controller_name] = v

        for tag_name, scalar in scalars.items():
            tf_writer.add_scalars(tag_name, scalar, step)

        tf_writer.flush()
