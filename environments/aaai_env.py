from pathlib import Path

from environments.sumo_env import SumoEnv, SumoEnvRunner
from settings import PROJECT_ROOT

from gym import error, spaces, utils
import traci
import traci.constants as tc

TRAFFIC_MOVEMENTS = 12
TRAFFICLIGHTS_PHASES = 8
LIGHT_DURATION = 10


class AaaiEnvRunner(SumoEnvRunner):
    def __init__(self, sumo_cmd, vehicle_generator_config, traffic_movements, traffic_lights_phases, light_duration):
        super().__init__(sumo_cmd, vehicle_generator_config)

        self.observation_space = spaces.Space(shape=(traffic_movements + 1,))
        self.action_space = spaces.Discrete(traffic_lights_phases)
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


class AaaiEnv(SumoEnv):
    def _instantiate_runner(self, sumo_cmd) -> SumoEnvRunner:
        return AaaiEnvRunner(
            sumo_cmd,
            self.vehicle_generator_config,
            self.traffic_movements,
            self.traffic_lights_phases,
            self.light_duration
        )

    def __init__(self, sumocfg_file_path, vehicle_generator_config,
                 traffic_movements=TRAFFIC_MOVEMENTS,
                 traffic_lights_phases=TRAFFICLIGHTS_PHASES,
                 light_duration=LIGHT_DURATION):
        super().__init__(sumocfg_file_path, vehicle_generator_config)
        self.traffic_movements = traffic_movements
        self.traffic_lights_phases = traffic_lights_phases
        self.light_duration = light_duration
