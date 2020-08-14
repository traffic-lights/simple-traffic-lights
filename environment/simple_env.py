from pathlib import Path

from environment.env import SumoEnv
from settings import PROJECT_ROOT
from random import randrange

import gym
from gym import error, spaces, utils
import numpy as np
import traci
import traci.constants as tc

NET_WIDTH = 200
NET_HEIGHT = 200
VEHICLE_LENGTH = 5
DIM_W = int(NET_WIDTH / VEHICLE_LENGTH)
DIM_H = int(NET_HEIGHT / VEHICLE_LENGTH)

DEFAULT_SPAWN_PERIOD = 10
MAX_LANE_OCCUPANCY = 0.6

DEFAULT_DURATION = 20.0
MIN_DURATION = 5.0
MAX_DURATION = 60.0

TRAFFICLIGHTS_PHASES = 4


class SimpleEnv(SumoEnv):
    def __init__(
            self,
            config_file=Path(PROJECT_ROOT, "environment", "2lane.sumocfg"),
            replay_folder=Path(PROJECT_ROOT, "replays"),
            save_replay=False,
            render=False,
    ):
        super().__init__(
            config_file=config_file,
            replay_folder=replay_folder,
            save_replay=save_replay,
            render=render
        )

        self.observation_space = spaces.Space(shape=(2, DIM_H, DIM_W))

        self.actions = [(i * 2, 5) for i in range(TRAFFICLIGHTS_PHASES)]
        self.actions = self.actions + [(i * 2, -5) for i in range(TRAFFICLIGHTS_PHASES)]
        self.actions = self.actions + [(-1, None)]

        self.action_space = spaces.Discrete(len(self.actions))

        self.phases_durations = [DEFAULT_DURATION for _ in range(4)]

        self.throughput = 0
        self.travel_time = 0

        self.traveling_cars = {}
        self.tls_id = traci.trafficlight.getIDList()[0]

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

    def _reset(self):
        self.phases_durations = [DEFAULT_DURATION for _ in range(4)]
        self.travel_time = 0
        self.throughput = 0

    def _take_action(self, action):
        action_tuple = self.actions[action]
        penalted = False

        arrived_cars = set()

        accumulated_travel_time = 0

        step = 0

        phases_tested_cnt = 0
        prev_phase = -1
        while True:
            self._generate_vehicles()
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
                            self.phases_durations[phase_id] < MIN_DURATION
                            or self.phases_durations[phase_id] > MAX_DURATION
                    ):
                        penalted = True

                    self.phases_durations[phase_id] = max(
                        MIN_DURATION, min(self.phases_durations[phase_id], MAX_DURATION)
                    )  # clamp

                traci.trafficlight.setPhaseDuration(
                    self.tls_id, self.phases_durations[phase_id]
                )

            time = traci.simulation.getTime()

            if step % 10 == 0:
                if self.save_replay:
                    traci.gui.screenshot(
                        "View #0", self.temp_folder.name + f"/state_{time}.png"
                    )

            for car in traci.simulation.getDepartedIDList():
                self.traveling_cars[car] = time

            for car in traci.simulation.getArrivedIDList():
                arrived_cars.add(car)

                accumulated_travel_time += time - self.traveling_cars[car]
                del self.traveling_cars[car]

            prev_phase = current_phase
            traci.simulationStep()
            step += 1

        incomings = set()
        outgoings = set()

        for entry in traci.trafficlight.getControlledLinks(self.tls_id):
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

        self.throughput += len(arrived_cars)

        self.travel_time += accumulated_travel_time

        if penalted:
            reward = -pressure - 200
        else:
            reward = -pressure

        return reward, pressure

    def get_throughput(self):
        return self.throughput

    def get_travel_time(self):  # in seconds
        if self.throughput == 0:
            return 0

        return round(self.travel_time / self.throughput, 2)
