import os, sys

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from pathlib import Path

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


VEHICLE_LENGTH = 5
NET_WIDTH = 200
NET_HEIGHT = 200

DEFAULT_DURATION = 20.0

DIM_W = int(NET_WIDTH / VEHICLE_LENGTH)
DIM_H = int(NET_HEIGHT / VEHICLE_LENGTH)

TRAFFICLIGHTS_PHASES = 4

REPLAY_FPS = 8


class SumoEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        config_file=Path(PROJECT_ROOT, "environment", "2lane.sumocfg"),
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

        self.sumo_cmd = [sumo_binary, "-c", config_file, "--no-step-log", "true"]

        traci.start(self.sumo_cmd)

        self.tls_id = traci.trafficlight.getIDList()[0]

        self.observation_space = spaces.Space(shape=(2, DIM_H, DIM_W))

        self.actions = [(i * 2, 5) for i in range(TRAFFICLIGHTS_PHASES)]
        self.actions = self.actions + [(i * 2, -5) for i in range(TRAFFICLIGHTS_PHASES)]
        self.actions = self.actions + [(-1, None)]

        self.action_space = spaces.Discrete(len(self.actions))

        self.phases_durations = [DEFAULT_DURATION for _ in range(4)]

        self.save_replay = save_replay
        self.temp_folder = tempfile.TemporaryDirectory()
        self.replay_folder = replay_folder

    def step(self, action):
        reward = None

        reward, info, penalted = self._take_action(action)
        state = self._snap_state()

        if penalted:
            return state, -100, True, info

        if traci.simulation.getMinExpectedNumber() == 0:
            return state, reward, True, info
        else:
            return state, reward, False, info

    def _snap_state(self):
        state = np.zeros((2, DIM_H + 1, DIM_W + 1))

        for vehicle in traci.vehicle.getIDList():
            traci.vehicle.subscribe(vehicle, (tc.VAR_POSITION, tc.VAR_SPEED))
            subscription_results = traci.vehicle.getSubscriptionResults(vehicle)

            vehicle_position = subscription_results[tc.VAR_POSITION]
            vehicle_speed = subscription_results[tc.VAR_SPEED]

            vehicle_discrete_position = (
                round(vehicle_position[0] / VEHICLE_LENGTH),
                round(vehicle_position[1] / VEHICLE_LENGTH),
            )

            state[0, vehicle_discrete_position[0], vehicle_discrete_position[1]] = 1
            state[1, vehicle_discrete_position[0], vehicle_discrete_position[1]] = int(
                round(vehicle_speed)
            )

        return state

    def _take_action(self, action):
        action_tuple = self.actions[action]
        penalted = False

        step = 0

        phases_tested_cnt = 0
        prev_phase = -1
        while traci.simulation.getMinExpectedNumber() > 0:
            current_phase = traci.trafficlight.getPhase(self.tls_id)

            if prev_phase != current_phase:
                phases_tested_cnt += 1

            if phases_tested_cnt == TRAFFICLIGHTS_PHASES * 2 + 1:
                break

            if current_phase % 2 == 0 and current_phase != prev_phase:
                phase_id = int(current_phase / 2)
                if current_phase == action_tuple[0]:
                    self.phases_durations[phase_id] += action_tuple[1]

                    if (
                        self.phases_durations[phase_id] < 0
                        or self.phases_durations[phase_id] > 60
                    ):
                        penalted = True

                    self.phases_durations[phase_id] = max(
                        0.0, min(self.phases_durations[phase_id], 60.0)
                    )  # clamp

                traci.trafficlight.setPhaseDuration(
                    self.tls_id, self.phases_durations[phase_id]
                )

            prev_phase = current_phase
            traci.simulationStep()

            step += 1
            if step % 10 == 0:
                if self.save_replay:
                    time = traci.simulation.getTime()
                    traci.gui.screenshot(
                        "View #0", self.temp_folder.name + f"/state_{time}.png"
                    )

        incomings = set()
        outgoings = set()

        for tls in traci.trafficlight.getIDList():
            for entry in traci.trafficlight.getControlledLinks(tls):
                if entry:
                    entry_tuple = entry[0]
                    if entry_tuple:
                        incomings.add(entry_tuple[0])
                        outgoings.add(entry_tuple[1])

        incomings_sum = 0
        outgoings_sum = 0

        for incoming in incomings:
            incomings_sum += traci.lane.getLastStepVehicleNumber(incoming)

        for outgoing in outgoings:
            outgoings_sum += traci.lane.getLastStepVehicleNumber(outgoing)

        pressure = abs(incomings_sum - outgoings_sum)

        # print(f'Pressure: {pressure}, incomings: {incomings_sum}, outgoings: {outgoings_sum}') debug :D

        return -pressure, pressure, penalted

    def reset(self):
        traci.close()
        traci.start(self.sumo_cmd)

        self.phases_durations = [DEFAULT_DURATION for _ in range(4)]

        return self._snap_state()

    def render(self, mode="human"):
        pass

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

    def close(self):
        traci.close()

        if self.save_replay:
            self._generate_gif()
            self.temp_folder.cleanup()
