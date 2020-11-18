import json
from collections import defaultdict
from pathlib import Path

from environments.sumo_env import SumoEnv
from settings import PROJECT_ROOT
import numpy as np

def evaluate_controllers_dict(runner, controllers, max_ep_len):
    states = runner.reset()
    ep_len = 0
    done = False
    all_rewards = []

    while not done:
        action = controller(state)

        actions = {}
        for tls_id, controller in controllers.items():
            state = states.get(tls_id)
            if not state:
                state = [0]*13
            actions[tls_id] = controller(state)

        states, rewards, done, info = runner.step(actions)
        all_rewards.extend(list(rewards.values()))

        ep_len += 1

    return {
        'throughput': runner.get_throughput(),
        'travel_time': runner.get_travel_time(),
        'mean_reward': np.mean(all_rewards)
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

    def _evaluate_traffic_controllers_dict(self, traffic_controllers_dict):

        metrics = {}

        for env_name, env in self.environments.items():
            print('evaluating %s' % env_name)
            with env.create_runner(render=True) as runner:
                controllers_with_connection = {junction: controller.with_connection(runner.connection) for
                                               junction, controller in traffic_controllers_dict.items()}

                metrics[env_name] = evaluate_controllers_dict(runner, controllers_with_connection, max_ep_len=300)

                print('res: ', metrics[env_name])

        return metrics

    def evaluate_all_dicts(self, traffic_controllers_dicts):

        if not isinstance(traffic_controllers_dicts, list):
            traffic_controllers_dicts = [traffic_controllers_dicts]

        all_metrics = []
        for i, traffic_controllers_dict in enumerate(traffic_controllers_dicts):
            print('evaluating set %d' % i)
            all_metrics.append(self._evaluate_traffic_controllers_dict(traffic_controllers_dict))

        return all_metrics
    '''
    def evaluate_to_tensorboard(self, traffic_controllers_dict, tf_writer, step):
        controller_names = list(traffic_controllers_dict.keys())
        controllers = list(traffic_controllers_dict.values())

        metrics = self._evaluate_traffic_controllers_dict(controllers)

        scalars = defaultdict(dict)

        for controller_name, metric in zip(controller_names, metrics):
            for env_name, evaluated in metric.items():
                for k, v in evaluated.items():
                    scalars[env_name + '/' + k][controller_name] = v

        for tag_name, scalar in scalars.items():
            tf_writer.add_scalars(tag_name, scalar, step)

        tf_writer.flush()
    '''
