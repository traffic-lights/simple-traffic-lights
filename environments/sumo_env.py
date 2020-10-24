import json
import os, sys
import uuid
from pathlib import Path

import gym

from settings import ENVIRONMENTS_FOLDER

DEFAULT_SUMO_PATH = os.path.join("/usr", "share", "sumo")
if "SUMO_HOME" not in os.environ:
    print("sumo home not in path")
    tools = os.path.join(DEFAULT_SUMO_PATH, "tools")
    os.environ['SUMO_HOME'] = DEFAULT_SUMO_PATH
else:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")

sys.path.append(tools)
import traci

from generators.vehicles_generator import VehiclesGenerator


class SumoEnvRunner(gym.Env):
    def __init__(self, sumo_cmd, vehicle_generator_config):
        self.sumo_cmd = sumo_cmd
        self.unique_id = uuid.uuid4()
        traci.start(self.sumo_cmd, label=self.unique_id)
        self.vehicle_generator = VehiclesGenerator.from_config_dict(vehicle_generator_config)

    def step(self, action):
        reward, info = self._take_action(action)
        state = self._snap_state()

        if traci.simulation.getMinExpectedNumber() == 0:
            return state, reward, True, info
        else:
            return state, reward, False, info

    def take_traci_control(self):
        traci.switch(self.unique_id)

    def reset(self):
        traci.close()
        traci.start(self.sumo_cmd)
        self._reset()

        self.vehicle_generator.reset()

        return self._snap_state()

    def render(self, mode='human'):
        pass

    def close(self):
        self.take_traci_control()
        traci.close()

    def _generate_vehicles(self):
        current_time = traci.simulation.getTime()
        self.vehicle_generator.generate_vehicles(current_time)

    def _snap_state(self):
        raise NotImplementedError

    def _take_action(self, action):
        raise NotImplementedError

    def _reset(self):
        raise NotImplementedError


class SumoEnv:
    def __init__(self, sumocfg_file_path, vehicle_generator_config):
        super().__init__()

        self.vehicle_generator_config = vehicle_generator_config
        self.sumocfg_file_path = sumocfg_file_path

    @staticmethod
    def from_config_file(file_path):
        from environments import ENVIRONMENTS_TYPE_MAPPER

        with open(file_path) as f:
            config_dict = json.load(f)

        env_config = config_dict['environment']
        env_type = env_config['type']
        config_file_path = Path(ENVIRONMENTS_FOLDER, config_dict['config_file'])
        return ENVIRONMENTS_TYPE_MAPPER[env_type](
            sumocfg_file_path=config_file_path,
            vehicle_generator_config=config_dict['vehicle_generator'],
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
            self.sumocfg_file_path,
            "--no-step-log",
            "true",
            "--time-to-teleport",
            "-1",
            "--max-depart-delay",
            "0"
        ]

        return self._instantiate_runner(sumo_cmd)

    def _instantiate_runner(self, sumo_cmd) -> SumoEnvRunner:
        raise NotImplementedError
