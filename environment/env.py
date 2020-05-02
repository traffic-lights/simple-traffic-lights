import gym
from gym import error, spaces, utils

import traci
import traci.constants as tc

import os, sys
import numpy as np


VEHICLE_LENGTH = 5
NET_WIDTH = 200
NET_HEIGHT = 200

DIM_W = int(NET_WIDTH / VEHICLE_LENGTH)
DIM_H = int(NET_HEIGHT / VEHICLE_LENGTH)

TRAFFICLIGHTS_PHASES = 4

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


class SumoEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, config_file="environment/2lane.sumocfg", render=False):
        super().__init__()

        sumo_binary = ""
        if render:
            sumo_binary = "sumo-gui"
        else:
            sumo_binary = "sumo"

        self.sumo_cmd = [sumo_binary, "-c", config_file]

        traci.start(self.sumo_cmd)

        self.tls_id = traci.trafficlight.getIDList()[0]

        self.observation_space = spaces.Space(
            shape=(DIM_W, DIM_H, 2)
        )  # Shape or something else?

        self.actions = [(i * 2, 5) for i in range(TRAFFICLIGHTS_PHASES)]
        self.actions = self.actions + [(i * 2, -5) for i in range(TRAFFICLIGHTS_PHASES)]
        self.actions = self.actions + [(-1, None)]

        self.action_space = spaces.Discrete(len(self.actions))

        self.phases_durations = [20.0, 20.0, 20.0, 20.0]

        self.last_wait_time = 0

    def step(self, action):
        reward = None

        reward = self._take_action(action)
        state = self._snap_state()

        if traci.simulation.getMinExpectedNumber() == 0:
            return state, reward, True
        else:
            return state, reward, False

    def _snap_state(self):
        state = np.zeros((DIM_W, DIM_H, 2))

        for vehicle in traci.vehicle.getIDList():
            traci.vehicle.subscribe(vehicle, (tc.VAR_POSITION, tc.VAR_SPEED))
            subscription_results = traci.vehicle.getSubscriptionResults(vehicle)

            vehicle_position = subscription_results[tc.VAR_POSITION]
            vehicle_speed = subscription_results[tc.VAR_SPEED]

            vehicle_discrete_position = (
                int(vehicle_position[0] / VEHICLE_LENGTH),
                int(vehicle_position[1] / 5),
            )

            state[vehicle_discrete_position] = [1, int(round(vehicle_speed))]

        # state = np.reshape(state, [-1, DIM_W * DIM_H, 2])

        return state

    def _take_action(self, action):
        action_tuple = self.actions[action]
        not_viable_action_penalty = 0

        step = 0
        wait_time_map = {}

        phases_tested_cnt = 0
        prev_phase = -1
        while traci.simulation.getMinExpectedNumber() > 0:
            if phases_tested_cnt == TRAFFICLIGHTS_PHASES * 2:
                break

            current_phase = traci.trafficlight.getPhase(self.tls_id)
            if current_phase % 2 == 0 and current_phase != prev_phase:
                phase_id = int(current_phase / 2)
                if current_phase == action_tuple[0]:
                    self.phases_durations[phase_id] += action_tuple[1]

                    if (
                        self.phases_durations[phase_id] < 0
                        or self.phases_durations[phase_id] > 60
                    ):
                        not_viable_action_penalty = -1000

                    self.phases_durations[phase_id] = max(
                        0.0, min(self.phases_durations[phase_id], 60.0)
                    )  # clamp

                traci.trafficlight.setPhaseDuration(
                    self.tls_id, self.phases_durations[phase_id]
                )

            if prev_phase != current_phase:
                phases_tested_cnt += 1

            prev_phase = current_phase
            traci.simulationStep()

            step += 1
            if step % 10 == 0:
                for vehicle in traci.vehicle.getIDList():
                    traci.vehicle.subscribe(vehicle, (tc.VAR_ACCUMULATED_WAITING_TIME,))
                    subscription_results = traci.vehicle.getSubscriptionResults(vehicle)

                    vehicle_wait_time = subscription_results[
                        tc.VAR_ACCUMULATED_WAITING_TIME
                    ]

                    wait_time_map[vehicle] = vehicle_wait_time

        wait_time_sum = 0
        for entry in wait_time_map:
            wait_time_sum += wait_time_map[entry]

        reward = self.last_wait_time - wait_time_sum

        self.last_wait_time = wait_time_sum

        return reward + not_viable_action_penalty

    def reset(self):
        traci.close()
        traci.start(self.sumo_cmd)

        return self._snap_state()

    def render(self, mode="human"):
        pass

    def save_simulation(self, path="sim_res.sbx"): # for future usage
        traci.simulation.saveState(path)

    def close(self):
        traci.close()
