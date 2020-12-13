from pathlib import Path

from environments.sumo_env import SumoEnv, SumoEnvRunner
from settings import PROJECT_ROOT
from random import randrange

import gym
from gym import error, spaces, utils
import numpy as np
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


class SimpleEnvRunner(SumoEnvRunner):
    def __init__(self,
                 sumo_cmd,
                 vehicle_generator_config,
                 max_steps=1500,
                 env_name=None):
        super().__init__(sumo_cmd, vehicle_generator_config, max_steps, env_name=env_name)

        self.observation_space = spaces.Space(shape=(2, DIM_H, DIM_W))

        self.actions = [(i * 2, 5) for i in range(TRAFFICLIGHTS_PHASES)]
        self.actions = self.actions + [(i * 2, -5) for i in range(TRAFFICLIGHTS_PHASES)]
        self.actions = self.actions + [(-1, None)]

        self.action_space = spaces.MultiDiscrete([len(self.actions)] * len([1]))

        self.phases_durations = [DEFAULT_DURATION for _ in range(4)]

        self.throughput = 0
        self.travel_time = 0

        self.traveling_cars = {}
        self.tls_id = self.connection.trafficlight.getIDList()[0]

    def _snap_state(self):
        state = np.zeros((2, DIM_H + 1, DIM_W + 1))

        for vehicle in self.connection.vehicle.getIDList():
            self.connection.vehicle.subscribe(vehicle, (tc.VAR_POSITION, tc.VAR_SPEED))
            subscription_results = self.connection.vehicle.getSubscriptionResults(vehicle)

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

        return np.asarray([state], dtype='float32')

    @staticmethod
    def dict_states_to_array(states):
        arr = []
        for state in states.values():
            arr.append(state)
        return arr

    def _take_action(self, action):
        action = action[0]
        action_tuple = self.actions[action]
        print(action_tuple)
        penalted = False

        arrived_cars = set()

        accumulated_travel_time = 0

        phases_tested_cnt = 0
        prev_phase = -1
        while True:
            self._generate_vehicles()
            current_phase = self.connection.trafficlight.getPhase(self.tls_id)

            if prev_phase != current_phase:
                phases_tested_cnt += 1

            if phases_tested_cnt == TRAFFICLIGHTS_PHASES * 2 + 1:
                break

            if current_phase % 2 == 0 and current_phase != prev_phase:
                phase_id = int(current_phase / 2)
                print(phase_id)
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

                self.connection.trafficlight.setPhaseDuration(
                    self.tls_id, self.phases_durations[phase_id]
                )

            time = self.connection.simulation.getTime()

            for car in self.connection.simulation.getDepartedIDList():
                self.traveling_cars[car] = time

            for car in self.connection.simulation.getArrivedIDList():
                arrived_cars.add(car)

                accumulated_travel_time += time - self.traveling_cars[car]
                del self.traveling_cars[car]

            prev_phase = current_phase
            self.connection.simulationStep()
            if self.connection.simulation.getTime() > self.max_steps:
                done = True
            else:
                done = False

        incomings = set()
        outgoings = set()

        for entry in self.connection.trafficlight.getControlledLinks(self.tls_id):
            if entry:
                entry_tuple = entry[0]
                if entry_tuple:
                    incomings.add(entry_tuple[0])
                    outgoings.add(entry_tuple[1])

        incomings_sum = 0
        outgoings_sum = 0

        for incoming in incomings:
            incomings_sum += self.connection.lane.getLastStepVehicleNumber(incoming)

        for outgoing in outgoings:
            outgoings_sum += self.connection.lane.getLastStepVehicleNumber(outgoing)

        pressure = abs(incomings_sum - outgoings_sum)

        self.throughput += len(arrived_cars)

        self.travel_time += accumulated_travel_time

        if penalted:
            reward = -pressure - 200
        else:
            reward = -pressure

        return done, reward, {
            'reward': [reward],
            'pressure': [pressure],
            'travel_time': self.get_travel_time(),
            'throughput': self.get_throughput()
        }

    def _reset(self):
        self.phases_durations = [DEFAULT_DURATION for _ in range(4)]
        self.travel_time = 0
        self.throughput = 0

    def get_throughput(self):
        return self.throughput

    def get_travel_time(self):  # in seconds
        if self.throughput == 0:
            return 0

        return round(self.travel_time / self.throughput, 2)


class SimpleEnv(SumoEnv):
    def _instantiate_runner(self, sumo_cmd) -> SumoEnvRunner:
        return SimpleEnvRunner(
            sumo_cmd,
            self.vehicle_generator_config,
            self.max_steps,
            self.env_name
        )

    def __init__(self, sumocfg_file_path, vehicle_generator_config,
                 max_steps=1500, env_name=None):
        super().__init__(sumocfg_file_path, vehicle_generator_config, max_steps, env_name=env_name)
