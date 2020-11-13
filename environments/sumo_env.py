import json
import multiprocessing
import uuid
from pathlib import Path

import gym

from settings import ENVIRONMENTS_FOLDER, init_sumo_tools
from generators.vehicles_generator import VehiclesGenerator

init_sumo_tools()
import traci

semaphore = multiprocessing.Semaphore(1)


class SumoEnvRunner(gym.Env):
    def __init__(self,
                 sumo_cmd,
                 vehicle_generator_config,
                 max_steps=None,
                 env_name=None
                 ):
        self.max_steps = max_steps
        self.sumo_cmd = sumo_cmd
        self.env_name = env_name
        with semaphore:
            self.unique_id = str(uuid.uuid4())
            traci.start(self.sumo_cmd, label=self.unique_id)
            if 'sumo' in self.sumo_cmd:
                self.sumo_cmd.remove('sumo')
            else:
                self.sumo_cmd.remove('sumo-gui')
            self.connection = traci.getConnection(self.unique_id)
        self.vehicle_generator = VehiclesGenerator.from_config_dict(self.connection, vehicle_generator_config)

        self.num_steps = 0
        self.was_step = False

    def step(self, action):
        self.num_steps += 1
        self.was_step = True
        reward, info = self._take_action(action)
        state = self._snap_state()
        if self.max_steps is not None and self.num_steps >= self.max_steps:
            done = True
        else:
            done = False

        info['env_name'] = self.env_name
        return state, reward, done, info

    def reset(self):
        if self.was_step:
            self.connection.load(self.sumo_cmd)
        self.was_step = False
        self._reset()

        self.vehicle_generator.reset()
        self.num_steps = 0
        return self._snap_state()

    def render(self, mode='human'):
        pass

    def close(self):
        self.connection.close()

    def _generate_vehicles(self):
        current_time = self.connection.simulation.getTime()
        self.vehicle_generator.generate_vehicles(current_time)

    def _snap_state(self):
        raise NotImplementedError

    def _take_action(self, action):
        raise NotImplementedError

    def _reset(self):
        raise NotImplementedError


class SumoEnv:
    def __init__(self, sumocfg_file_path, vehicle_generator_config, max_steps=None, env_name=None):
        super().__init__()
        self.max_steps = max_steps
        self.vehicle_generator_config = vehicle_generator_config
        self.sumocfg_file_path = sumocfg_file_path
        self.env_name = env_name

    @staticmethod
    def from_config_file(file_path, max_steps=None, env_name=None):
        from environments import ENVIRONMENTS_TYPE_MAPPER

        with open(file_path) as f:
            config_dict = json.load(f)
        env_config = config_dict['environment']
        env_type = env_config['type']
        config_file_path = Path(ENVIRONMENTS_FOLDER, config_dict['config_file'])
        return ENVIRONMENTS_TYPE_MAPPER[env_type](
            sumocfg_file_path=config_file_path,
            vehicle_generator_config=config_dict['vehicle_generator'],
            max_steps=max_steps,
            env_name=env_name,
            **env_config['additional_params'] if 'additional_params' else {}
        )

    def create_runner(self, render=False) -> SumoEnvRunner:
        if render:
            sumo_binary = "sumo-gui"
        else:
            sumo_binary = "sumo"

        sumo_cmd = [
            sumo_binary,
            "-c",
            str(self.sumocfg_file_path),
            "--no-step-log",
            "true",
            "--time-to-teleport",
            "-1",
            "--max-depart-delay",
            "10"
        ]

        return self._instantiate_runner(sumo_cmd)

    def _instantiate_runner(self, sumo_cmd) -> SumoEnvRunner:
        raise NotImplementedError
