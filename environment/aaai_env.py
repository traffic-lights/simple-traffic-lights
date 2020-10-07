from pathlib import Path

from environment.env import SumoEnv
from settings import PROJECT_ROOT
from environment.configs_loader import load_from_file

from gym import error, spaces, utils
import traci
import traci.constants as tc

TRAFFIC_MOVEMENTS = 12
TRAFFICLIGHTS_PHASES = 8
LIGHT_DURATION = 10


class AaaiEnv(SumoEnv):
    def __init__(self, save_replay=False, render=False, light_duration=LIGHT_DURATION, key="aaai_random"):
        env_configs = load_from_file(key)

        super().__init__(
            env_configs=env_configs, save_replay=save_replay, render=render
        )

        self.observation_space = spaces.Space(shape=(TRAFFIC_MOVEMENTS + 1,))
        self.action_space = spaces.Discrete(TRAFFICLIGHTS_PHASES)
        self.tls_id = traci.trafficlight.getIDList()[0]
        self.light_duration = light_duration
        self.previous_action = 0
        self.traveling_cars = {}

        self.travel_time = 0
        self.throughput = 0

    def _snap_state(self):
        pressures = [self.previous_action]

        for entry in traci.trafficlight.getControlledLinks(self.tls_id):
            if entry:
                entry_tuple = entry[0]
                if entry_tuple:
                    my_pressure = traci.lane.getLastStepVehicleNumber(
                        entry_tuple[0]
                    ) - traci.lane.getLastStepVehicleNumber(entry_tuple[1])
                    pressures.append(my_pressure)
        return pressures

    def _take_action(self, action):
        arrived_cars = set()

        accumulated_travel_time = 0

        # turn yellow light if different action

        if self.previous_action != action:
            traci.trafficlight.setPhase(self.tls_id, 2 * self.previous_action + 1)
            start_time = traci.simulation.getTime()
            dur = traci.trafficlight.getPhaseDuration(self.tls_id)
            while traci.simulation.getTime() - start_time < dur - 0.1:
                self._generate_vehicles()
                time = traci.simulation.getTime()

                for car in traci.simulation.getDepartedIDList():
                    self.traveling_cars[car] = time

                for car in traci.simulation.getArrivedIDList():
                    arrived_cars.add(car)

                    accumulated_travel_time += time - self.traveling_cars[car]
                    del self.traveling_cars[car]

                traci.simulationStep()

        self.previous_action = action

        traci.trafficlight.setPhase(self.tls_id, 2 * action)
        traci.trafficlight.setPhaseDuration(self.tls_id, self.light_duration)

        start_time = traci.simulation.getTime()
        while traci.simulation.getTime() - start_time < self.light_duration - 0.1:
            self._generate_vehicles()
            time = traci.simulation.getTime()

            for car in traci.simulation.getDepartedIDList():
                self.traveling_cars[car] = time

            for car in traci.simulation.getArrivedIDList():
                arrived_cars.add(car)

                accumulated_travel_time += time - self.traveling_cars[car]
                del self.traveling_cars[car]

            traci.simulationStep()

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
        reward = -pressure
        return reward, pressure

    def _reset(self):
        self.travel_time = 0
        self.throughput = 0

    def get_throughput(self):
        return self.throughput

    def get_travel_time(self):  # in seconds
        if self.throughput == 0:
            return 0

        return round(self.travel_time / self.throughput, 2)
