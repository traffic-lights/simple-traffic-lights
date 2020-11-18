import numpy as np
from gym import spaces

from environments.sumo_env import SumoEnv, SumoEnvRunner

TRAFFIC_MOVEMENTS = 12
TRAFFICLIGHTS_PHASES = 8
LIGHT_DURATION = 10


class AaaiEnvRunner(SumoEnvRunner):
    def __init__(self,
                 sumo_cmd,
                 vehicle_generator_config,
                 junctions,
                 traffic_movements,
                 traffic_lights_phases,
                 light_duration,
                 max_steps=1500,
                 env_name=None):
        super().__init__(sumo_cmd, vehicle_generator_config, max_steps, env_name=env_name)

        self.junctions = junctions
        self.traffic_lights_phases = traffic_lights_phases

        self.observation_space = spaces.Space(shape=(len(junctions)*(traffic_movements + 1),))
        self.action_space = spaces.MultiDiscrete([traffic_lights_phases] * len(junctions))

        self.light_duration = light_duration

        self.previous_actions = {junction: 0 for junction in junctions}
        self.traveling_cars = {}

        self.travel_time = 0
        self.throughput = 0

        self.restarted = True

    def dict_states_to_array(self, states):
        arr = []
        for state in states.values():
            arr.extend(state)
        return arr

    def arr_states_to_dict(self, states):
        dct = {}
        for i, junction in enumerate(self.junctions):
            dct[junction] = states[13*i:13*(i+1)]
        return dct

    def dict_actions_to_val(self, actions):
        val = 0
        for action in actions.values():
            val *= self.traffic_lights_phases
            val += action
        return val

    def val_action_to_dict(self, val):
        reversed_arr = []
        for _ in range(len(self.junctions)):
            reversed_arr.append(int(val % self.traffic_lights_phases))
            val //= self.traffic_lights_phases

        dct = dict(zip(self.junctions, reversed(reversed_arr)))
        return dct

    def _snap_state(self):
        states = {
            junction: [0]*13 for junction in self.junctions
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

        return np.asarray(self.dict_states_to_array(states), dtype='float32')

    def _simulate_step(self):
        arrived_cars = set()

        accumulated_travel_time = 0
        if self.connection.simulation.getTime() > self.max_steps:
            return True, arrived_cars, accumulated_travel_time

        self._generate_vehicles()
        time = self.connection.simulation.getTime()

        for car in self.connection.simulation.getDepartedIDList():
            self.traveling_cars[car] = time

        if not self.restarted:
            for car in self.connection.simulation.getArrivedIDList():
                arrived_cars.add(car)

                accumulated_travel_time += time - self.traveling_cars[car]
                del self.traveling_cars[car]

        self.connection.simulationStep()

        self.restarted = False
        return False, arrived_cars, accumulated_travel_time

    def _proceed_actions(self, phase_changes, phase_shift_f):
        arrived_cars = set()

        accumulated_travel_time = 0

        for tls_id, phase in phase_changes.items():
            self.connection.trafficlight.setPhase(tls_id, phase_shift_f(phase))

        start_time = self.connection.simulation.getTime()
        dur = self.connection.trafficlight.getPhaseDuration(next(iter(phase_changes.keys())))
        done = False
        while not done and self.connection.simulation.getTime() - start_time < dur - 0.1:
            done, my_cars, my_time = self._simulate_step()
            arrived_cars |= my_cars
            accumulated_travel_time += my_time

        return arrived_cars, accumulated_travel_time

    def _take_action(self, action):
        actions = dict(zip(self.junctions, action))

        arrived_cars = set()

        accumulated_travel_time = 0

        rewards = {}
        pressures = {}

        def yellow(act): return 3 * act + 1
        def red(act): return 3 * act + 2

        # turn yellow and red light if different action

        if not self.restarted:

            phase_changes = {}

            for tls_id, act in actions.items():
                if act != self.previous_actions[tls_id]:
                    phase_changes[tls_id] = self.previous_actions[tls_id]

            if phase_changes:
                # yellows
                my_cars, my_time = self._proceed_actions(phase_changes, yellow)
                arrived_cars |= my_cars
                accumulated_travel_time += my_time
                # reds
                my_cars, my_time = self._proceed_actions(phase_changes, red)
                arrived_cars |= my_cars
                accumulated_travel_time += my_time

        self.previous_actions = actions

        for tls_id, act in actions.items():
            self.connection.trafficlight.setPhase(tls_id, 3 * act)
            self.connection.trafficlight.setPhaseDuration(tls_id, self.light_duration)

        start_time = self.connection.simulation.getTime()
        done = False
        while not done and self.connection.simulation.getTime() - start_time < self.light_duration - 0.1:
            done, my_cars, my_time = self._simulate_step()
            arrived_cars |= my_cars
            accumulated_travel_time += my_time

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

        return done, np.mean(list(rewards.values())), {
            'reward': rewards,
            'pressure': pressures,
            'travel_time': self.get_travel_time(),
            'throughput': self.get_throughput()
        }

    def _reset(self):
        self.travel_time = 0
        self.throughput = 0
        self.previous_actions = {junction: 0 for junction in self.junctions}
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
            self.junctions,
            self.traffic_movements,
            self.traffic_lights_phases,
            self.light_duration,
            self.max_steps,
            self.env_name
        )

    def __init__(self, sumocfg_file_path, vehicle_generator_config,
                 max_steps=1500, env_name=None,
                 junctions=None,
                 traffic_movements=TRAFFIC_MOVEMENTS,
                 traffic_lights_phases=TRAFFICLIGHTS_PHASES,
                 light_duration=LIGHT_DURATION):
        super().__init__(sumocfg_file_path, vehicle_generator_config, max_steps, env_name=env_name)
        if junctions is None:
            junctions = ['gneJ18']
        self.junctions = junctions
        self.traffic_movements = traffic_movements
        self.traffic_lights_phases = traffic_lights_phases
        self.light_duration = light_duration
