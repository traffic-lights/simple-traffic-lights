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
import re
from os import listdir
from os.path import isfile, join
import tempfile
import numpy as np
import datetime

from settings import PROJECT_ROOT

DEFAULT_SPAWN_PERIOD = 10
MAX_LANE_OCCUPANCY = 0.6

REPLAY_FPS = 8


class Lane:
    def __init__(self, lane_id, route_id, spawn_period=DEFAULT_SPAWN_PERIOD):
        super().__init__()

        assert spawn_period >= 0, 'negative spawn period'

        edge_id = traci.lane.getEdgeID(lane_id)
        # lane index from laneID
        self.lane_id = lane_id
        self.index = int(re.sub(f'{edge_id}_', "", lane_id))
        self.route_id = route_id
        self.spawn_period = spawn_period

        self.next_timer = 0

    def add_car(self, current_time, car_ids):
        last_step_occupancy = traci.lane.getLastStepOccupancy(self.lane_id)
        if (
                self.next_timer - current_time <= 0
                and last_step_occupancy <= MAX_LANE_OCCUPANCY
        ):
            car_id = f"{self.route_id}_{car_ids}"
            traci.vehicle.add(
                vehID=car_id, routeID=self.route_id, departLane=self.index
            )
            self.next_timer = current_time + self.spawn_period

            return True
        return False

    def reset_spawning_data(self):
        self.next_timer = 0
        self.car_ids = 0

    def __str__(self):
        return f"lane: {self.lane_id} route: {self.route_id}"


class SumoEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
            self,
            config_file=Path(PROJECT_ROOT, "environment", "2lane/2lane.sumocfg"),
            replay_folder=Path(PROJECT_ROOT, "replays"),
            save_replay=False,
            render=False,
    ):
        super().__init__()

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

        self.start_lanes = []
        self.car_ids = {}

        self.routes = traci.route.getIDList()

        for lane in traci.lane.getIDList():
            start_edge = traci.lane.getEdgeID(lane)
            links = traci.lane.getLinks(lane)

            spawn_period = traci.lane.getParameter(lane, "period")
            try:
                spawn_period = int(spawn_period)
            except ValueError:
                if spawn_period != "None" and spawn_period != "":
                    print(f'bad period value: {spawn_period}')

                spawn_period = None

            for link in links:
                end_edge = traci.lane.getEdgeID(link[0])
                for route in self.routes:
                    route_edges = traci.route.getEdges(route)
                    if start_edge in route_edges and end_edge in route_edges:
                        self.car_ids[route] = 0
                        if spawn_period is not None:
                            self.start_lanes.append(Lane(lane, route, spawn_period))

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

        for route in self.routes:
            self.car_ids[route] = 0

        for lane in self.start_lanes:
            lane.reset_spawning_data()

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
        for lane in self.start_lanes:
            if lane.add_car(current_time, self.car_ids[lane.route_id]):
                self.car_ids[lane.route_id] += 1
