from collections import namedtuple
from dataclasses import dataclass

import numpy as np

from environments.sumo_env import SumoEnv, SumoEnvRunner
from gym import error, spaces, utils

from traci import TraCIException

TRAFFIC_MOVEMENTS = 12
TRAFFICLIGHTS_PHASES = 8
LIGHT_DURATION = 10


class AaaiEnvRunner(SumoEnvRunner):
    def __init__(self,
                 sumo_cmd,
                 vehicle_generator_config,
                 traffic_movements,
                 traffic_lights_phases,
                 light_duration,
                 max_steps=None,
                 env_name=None):
        super().__init__(sumo_cmd, vehicle_generator_config, max_steps, env_name=env_name)
        self.observation_space = spaces.Space(shape=(traffic_movements + 1,))
        self.action_space = spaces.Discrete(traffic_lights_phases)
        self.tls_id = self.connection.trafficlight.getIDList()[0]
        self.light_duration = light_duration

        self.previous_action = 0
        self.traveling_cars = {}

        self.travel_time = 0
        self.throughput = 0

        self.restarted = True

    def _snap_state(self):
        pressures = [self.previous_action]

        for entry in self.connection.trafficlight.getControlledLinks(self.tls_id):
            if entry:
                entry_tuple = entry[0]
                if entry_tuple:
                    my_pressure = self.connection.lane.getLastStepVehicleNumber(
                        entry_tuple[0]
                    ) - self.connection.lane.getLastStepVehicleNumber(entry_tuple[1])
                    pressures.append(my_pressure)
        return np.array(pressures, dtype=np.float32)

    def _simulate_step(self):
        arrived_cars = set()

        accumulated_travel_time = 0
        accumulated_waiting_time = 0

        self._generate_vehicles()
        time = self.connection.simulation.getTime()

        for car in self.connection.simulation.getDepartedIDList():
            self.traveling_cars[car] = time

        if not self.restarted:
            for car in self.connection.simulation.getArrivedIDList():
                arrived_cars.add(car)

                accumulated_travel_time += time - self.traveling_cars[car]
                # try:
                #     print(car, self.connection.vehicle.getAccumulatedWaitingTime(car))
                # except TraCIException as e:
                #     pass
                del self.traveling_cars[car]

        # for l_id in self.vehicle_generator.lanes.keys():
        #     print("lane id: {}, time: {}".format(l_id, self.connection.lane.getWaitingTime(l_id)))

        # my_travel_time = sum([self.connection.lane.getTraveltime(l_id) for l_id in self.vehicle_generator.lanes.keys()])

        # print(my_travel_time, accumulated_travel_time)

        self.connection.simulationStep()
        self.restarted = False
        return arrived_cars, accumulated_travel_time

    def _take_action(self, action):
        arrived_cars = set()

        accumulated_travel_time = 0

        # turn yellow light if different action

        if self.previous_action != action and not self.restarted:
            self.connection.trafficlight.setPhase(self.tls_id, 2 * self.previous_action + 1)
            start_time = self.connection.simulation.getTime()
            dur = self.connection.trafficlight.getPhaseDuration(self.tls_id)
            while self.connection.simulation.getTime() - start_time < dur - 0.1:
                my_cars, my_time = self._simulate_step()
                arrived_cars |= my_cars
                accumulated_travel_time += my_time

        self.previous_action = action

        self.connection.trafficlight.setPhase(self.tls_id, 2 * action)
        self.connection.trafficlight.setPhaseDuration(self.tls_id, self.light_duration)

        start_time = self.connection.simulation.getTime()
        while self.connection.simulation.getTime() - start_time < self.light_duration - 0.1:
            my_cars, my_time = self._simulate_step()
            arrived_cars |= my_cars
            accumulated_travel_time += my_time

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
        reward = -pressure

        return reward, {
            'reward': reward,
            'pressure': pressure,
            'travel_time': self.get_travel_time(),
            'throughput': self.get_throughput()
        }

    def _reset(self):
        self.travel_time = 0
        self.throughput = 0
        self.previous_action = 0
        self.traveling_cars = {}
        self.restarted = True

    def get_throughput(self):
        return self.throughput

    def get_travel_time(self):  # in seconds
        if self.throughput == 0:
            return 0

        return round(self.travel_time / self.throughput, 2)


class AaaiEnv(SumoEnv):
    def _instantiate_runner(self, sumo_cmd) -> SumoEnvRunner:
        return AaaiEnvRunner(
            sumo_cmd,
            self.vehicle_generator_config,
            self.traffic_movements,
            self.traffic_lights_phases,
            self.light_duration,
            self.max_steps,
            self.env_name
        )

    def __init__(self, sumocfg_file_path, vehicle_generator_config,
                 max_steps=None, env_name=None,
                 traffic_movements=TRAFFIC_MOVEMENTS,
                 traffic_lights_phases=TRAFFICLIGHTS_PHASES,
                 light_duration=LIGHT_DURATION):
        super().__init__(sumocfg_file_path, vehicle_generator_config, max_steps, env_name=env_name)
        self.traffic_movements = traffic_movements
        self.traffic_lights_phases = traffic_lights_phases
        self.light_duration = light_duration
