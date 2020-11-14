from environments.sumo_env import SumoEnv, SumoEnvRunner
from gym import error, spaces, utils

TRAFFIC_MOVEMENTS = 12
TRAFFICLIGHTS_PHASES = 8
LIGHT_DURATION = 10


class AaaiEnvRunner(SumoEnvRunner):
    def __init__(self, sumo_cmd, vehicle_generator_config, traffic_movements, traffic_lights_phases,
                 light_duration):
        super().__init__(sumo_cmd, vehicle_generator_config)

        self.observation_space = spaces.Space(shape=(traffic_movements + 1,))
        self.action_space = spaces.Discrete(traffic_lights_phases)
        # self.tls_id = self.connection.trafficlight.getIDList()[0]
        self.light_duration = light_duration
        self.previous_actions = {}
        self.traveling_cars = {}

        self.travel_time = 0
        self.throughput = 0

    def _snap_state(self):
        states = {
            'gneJ25': [0]*13,
            'gneJ26': [0]*13,
            'gneJ27': [0]*13,
            'gneJ28': [0]*13,
        }
        for tls_id, action in self.previous_actions.items():
            pressures = [action]

            for entry in self.connection.trafficlight.getControlledLinks(tls_id):
                if entry:
                    entry_tuple = entry[0]
                    if entry_tuple:
                        my_pressure = self.connection.lane.getLastStepVehicleNumber(
                            entry_tuple[0]
                        ) - self.connection.lane.getLastStepVehicleNumber(entry_tuple[1])
                        pressures.append(my_pressure)

            states[tls_id] = pressures

        return states

    def _continue_simulation(self, arrived_cars, accumulated_travel_time):
        self._generate_vehicles()
        time = self.connection.simulation.getTime()

        for car in self.connection.simulation.getDepartedIDList():
            self.traveling_cars[car] = time

        for car in self.connection.simulation.getArrivedIDList():
            arrived_cars.add(car)

            accumulated_travel_time += time - self.traveling_cars[car]
            del self.traveling_cars[car]

        self.connection.simulationStep()

    def _take_action(self, actions):
        arrived_cars = set()

        accumulated_travel_time = 0

        rewards = {}
        pressures = {}

        # turn yellow light if different action

        if self.previous_actions:

            phase_changes = {}

            for tls_id, action in actions.items():
                if action != self.previous_actions[tls_id]:
                    phase_changes[tls_id] = self.previous_actions[tls_id]

            for tls_id, prev_action in phase_changes.items():
                self.connection.trafficlight.setPhase(tls_id, 2 * prev_action + 1)
                start_time = self.connection.simulation.getTime()
                dur = self.connection.trafficlight.getPhaseDuration(tls_id)

                while self.connection.simulation.getTime() - start_time < dur - 0.1:
                    self._continue_simulation(arrived_cars, accumulated_travel_time)

        # turn green

        self.previous_actions = actions

        for tls_id, action in actions.items():
            self.connection.trafficlight.setPhase(tls_id, 2 * action)
            self.connection.trafficlight.setPhaseDuration(tls_id, self.light_duration)

        start_time = self.connection.simulation.getTime()

        while self.connection.simulation.getTime() - start_time < self.light_duration - 0.1:
            self._continue_simulation(arrived_cars, accumulated_travel_time)

        for tls_id in actions.keys():

            incomings = set()
            outgoings = set()

            for entry in self.connection.trafficlight.getControlledLinks(tls_id):
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

            pressures[tls_id] = pressure
            rewards[tls_id] = -pressure

        return rewards, pressures

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
