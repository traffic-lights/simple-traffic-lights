import copy
import json
from collections import defaultdict
from pathlib import Path

from environments.sumo_env import SumoEnv
from settings import PROJECT_ROOT
import numpy as np
from multiprocessing import Process, Pipe


def evaluate_controller(runner, controllers, state_mean_std=None):
    states = runner.reset()
    ep_len = 0
    done = False
    all_rewards = []
    if not isinstance(controllers, list):
        controllers = [controllers] * len(runner.junctions)
    elif len(controllers) == 1:
        controllers = controllers * len(runner.junctions)

    while not done:
        actions = []
        for i, controller in enumerate(controllers):
            state = states[i]
            if state_mean_std is not None:
                state = (np.array(state) - state_mean_std[0]) / state_mean_std[1]

            if state is not None:
                actions.append(controller(state))
            else:
                actions.append(None)

        states, rewards, done, info = runner.step(actions)
        all_rewards.extend(info['reward'])
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
            test_case['name']: SumoEnv.from_config_file(Path(PROJECT_ROOT, test_case['config']))
            for test_case in self.test_cases_list
        }

        self.async_executions = []

    def evaluate_traffic_controllers(self, traffic_controllers, render=False, state_mean_std=None):
        """
        :param traffic_controllers: list of controllers
        :return: list of metrics for each controller, in the same order as given list of controllers
        """

        if not isinstance(traffic_controllers, list):
            traffic_controllers = [traffic_controllers]

        metrics = [{} for _ in traffic_controllers]

        for env_name, env in self.environments.items():
            with env.create_runner(render=render) as runner:
                for i_controller, controller in enumerate(traffic_controllers):
                    if not isinstance(controller, list):
                        controller = [controller]
                    controller = [c.with_connection(runner.connection) for c in controller]

                    metrics[i_controller][env_name] = evaluate_controller(runner, controller, state_mean_std)

        return metrics

    def evaluate_traffic_controllers_async(self, traffic_controllers, send):
        m = self.evaluate_traffic_controllers(traffic_controllers, False, None)
        send.send(m)

    def evaluate_to_tensorboard(self, traffic_controllers_dict, tf_writer, step, state_mean_std=None):
        controller_names = list(traffic_controllers_dict.keys())
        controllers = list(traffic_controllers_dict.values())

        metrics = self.evaluate_traffic_controllers(controllers, state_mean_std)

        self._write_metrics(tf_writer, controller_names, metrics, step)

    def _write_metrics(self, tf_writer, controller_names, metrics, step):
        scalars = defaultdict(dict)

        for controller_name, metric in zip(controller_names, metrics):
            for env_name, evaluated in metric.items():
                for k, v in evaluated.items():
                    scalars[env_name + '/' + k][controller_name] = v

        for tag_name, scalar in scalars.items():
            tf_writer.add_scalars(tag_name, scalar, step)

        tf_writer.flush()

    def evaluate_to_tensorboard_async(self, traffic_controllers_dict, tf_writer, step, state_mean_std=None):
        for rec, proc, controller_names, step in self.async_executions:
            # this will wait until each process has finished and then collect the data
            metrics = rec.recv()
            proc.join()

            self._write_metrics(tf_writer, controller_names, metrics, step)

        self.async_executions = []

        controller_names = list(traffic_controllers_dict.keys())
        controllers = [copy.deepcopy(c) for c in traffic_controllers_dict.values()]

        recv, send = Pipe()

        # gets process ready with function to complete and arguments
        pid = Process(target=self.evaluate_traffic_controllers_async, args=(controllers, send))

        # starts process
        pid.start()

        # keeps track of process
        self.async_executions.append((recv, pid, controller_names, step))
