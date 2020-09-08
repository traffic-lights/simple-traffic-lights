import os, sys

DEFAULT_SUMO_PATH = os.path.join("/usr", "share", "sumo")
if "SUMO_HOME" not in os.environ:
    tools = os.path.join(DEFAULT_SUMO_PATH, "tools")
else:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")

sys.path.append(tools)

from pathlib import Path

from random import randrange

import gym
from gym import error, spaces, utils

import traci
import traci.constants as tc

import imageio
from os import listdir
from os.path import isfile, join
import tempfile
import numpy as np
import datetime

from settings import PROJECT_ROOT

from environment.vehicles_generator import (
    SinusoidalGenerator,
    XMLGenerator,
    ConstGenerator,
)

REPLAY_FPS = 8


class SumoEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self, env_configs, save_replay=False, render=False,
    ):
        super().__init__()

        config_file = Path(PROJECT_ROOT, "environment", env_configs["config_file"])
        replay_folder = Path(PROJECT_ROOT, env_configs["replay_folder"])
        generator_type = env_configs["vehicle_generator"]["type"]

        sumo_binary = ""
        if render:
            sumo_binary = "sumo-gui"
        else:
            sumo_binary = "sumo"

        self.sumo_cmd = [
            sumo_binary,
            "-c",
            config_file,
            "--no-step-log",
            "true",
            "--time-to-teleport",
            "-1",
        ]

        traci.start(self.sumo_cmd)
        self.save_replay = save_replay
        self.temp_folder = tempfile.TemporaryDirectory()
        self.replay_folder = replay_folder

        if generator_type == "const":
            self.vehicle_generator = ConstGenerator
        elif generator_type == "sin":
            self.vehicle_generator = SinusoidalGenerator
            generator_lanes = env_configs["vehicle_generator"]["lanes"]
        else:
            print(f"{generator_type} unknown generator type")
            sys.exit(-1)

        generator_lanes = env_configs["vehicle_generator"]["lanes"]

        for lane in generator_lanes:
            self.vehicle_generator.add_lane(**lane)

    def step(self, action):
        reward, info = self._take_action(action)
        state = self._snap_state()

        if traci.simulation.getMinExpectedNumber() == 0:
            return state, reward, True, info
        else:
            return state, reward, False, info

    def _snap_state(self):
        raise NotImplementedError

    def _take_action(self, action):
        raise NotImplementedError

    def _reset(self):
        raise NotImplementedError

    def reset(self):
        traci.close()
        traci.start(self.sumo_cmd)
        self._reset()

        self.vehicle_generator.reset()

        return self._snap_state()

    def render(self, mode="human"):
        pass

    def close(self):
        traci.close()

        if self.save_replay:
            self._generate_gif()
            self.temp_folder.cleanup()

    def _generate_gif(self):
        src_path = self.temp_folder.name
        res_path = self.replay_folder

        onlyfiles = [f for f in listdir(src_path) if isfile(join(src_path, f))]
        filenames = [f for f in onlyfiles if ".png" in f]

        res_name = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        with imageio.get_writer(
            f"{res_path}/{res_name}_sim.gif", mode="I", fps=REPLAY_FPS
        ) as writer:
            for filename in filenames:
                image = imageio.imread(f"{src_path}/{filename}")
                writer.append_data(image)

    def _generate_vehicles(self):
        current_time = traci.simulation.getTime()
        self.vehicle_generator.generate_vehicles(current_time)
