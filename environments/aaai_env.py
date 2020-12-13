from enum import Enum
from dataclasses import dataclass

import numpy as np
from gym import spaces

from environments.sumo_env import SumoEnv, SumoEnvRunner

TRAFFIC_MOVEMENTS = 12
TRAFFICLIGHTS_PHASES = 8
LIGHT_DURATION = 10

class EventType(Enum):
    SWITCH_TO_GREEN = 0
    SWITCH_TO_RED = 1
    ASK_FOR_ACTION = 2

@dataclass
class Event:
    jun_intern_id: int
    event_type: EventType
    simulation_time: int

class AaaiEnvRunner(SumoEnvRunner):
    def __init__(self,
                 sumo_cmd,
                 vehicle_generator_config,
                 junctions,
                 traffic_movements,
                 traffic_lights_phases,
                 light_duration,
                 clusters,
                 max_steps=1500,
                 env_name=None):
        super().__init__(sumo_cmd, vehicle_generator_config, max_steps, env_name=env_name)

        if not clusters:
            clusters = {}

        self.junctions = junctions
        self.cluster_map = clusters
        self.traffic_lights_phases = traffic_lights_phases

        self.observation_space = spaces.Space(shape=(len(junctions), traffic_movements + 1))
        self.action_space = spaces.MultiDiscrete([traffic_lights_phases] * len(junctions))

        self.light_duration = light_duration

        self.previous_actions = {}
        self.clustered_juncions = {}
        for junction in self.junctions:
            cluster = self.cluster_map.get(junction)
            if cluster:
                for jun, _ in cluster["tls_to_phases"].items():
                    self.clustered_juncions[jun] = junction
                    self.previous_actions[jun] = (0, 1, 2, 3)
            else:
                self.previous_actions[junction] = (0, 1, 2, 3)

        self.traveling_cars = {}

        self.travel_time = 0
        self.throughput = 0

        self.green_dur = self.light_duration
        self.connection.trafficlight.setPhase(self.junctions[0], 1)
        self.yellow_dur = self.connection.trafficlight.getPhaseDuration(self.junctions[0])
        self.connection.trafficlight.setPhase(self.junctions[0], 2)
        self.red_dur = self.connection.trafficlight.getPhaseDuration(self.junctions[0])
        self.connection.trafficlight.setPhase(self.junctions[0], 0)

        self.curr_phases = [-1] * len(junctions)

        self.events = []
        self.ret_state = [True] * len(junctions)

        self.restarted = True

    @staticmethod
    def dict_states_to_array(states):
        arr = []
        for state in states.values():
            arr.append(state)
        return arr

    def arr_states_to_dict(self, states):
        dct = {}
        for i, junction in enumerate(self.junctions):
            dct[junction] = states[13 * i:13 * (i + 1)]
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

        states = {}
        for junction in self.junctions:
            cluster = self.cluster_map.get(junction)
            if cluster:
                states[junction] = [0] * (1 + len(cluster["traffic_movements"]))
            else:
                states[junction] = [0] * 13

        clustered_links = set()

        for tls_id, phases in self.previous_actions.items():

            pressures = [phases[0]]

            cluster = self.clustered_juncions.get(tls_id)

            if cluster:
                if cluster not in clustered_links:
                    clustered_links.add(cluster)
                    for incoming, outgoing in self.cluster_map[cluster]["traffic_movements"]:
                        pressures.append(self.connection.lane.getLastStepVehicleNumber(incoming) -
                                         self.connection.lane.getLastStepVehicleNumber(outgoing))

                    states[cluster] = pressures
            else:
                for entry in self.connection.trafficlight.getControlledLinks(tls_id):
                    if entry:
                        entry_tuple = entry[0]
                        if entry_tuple:
                            my_pressure = self.connection.lane.getLastStepVehicleNumber(
                                entry_tuple[0]
                            ) - self.connection.lane.getLastStepVehicleNumber(entry_tuple[1])
                            pressures.append(my_pressure)

                states[tls_id] = pressures

        ret_arr = self.dict_states_to_array(states)
        ret_val = []
        for ret, arr in zip(self.ret_state, ret_arr):
            if ret:
                ret_val.append(np.asarray(arr, dtype='float32'))
            else:
                ret_val.append(None)

        self.ret_state = [False] * len(self.junctions)
        return ret_val

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

    def _proceed_actions(self, phase_changes, phase_change_id):
        arrived_cars = set()

        accumulated_travel_time = 0

        for tls_id, phases in phase_changes.items():
            self.connection.trafficlight.setPhase(tls_id, phases[phase_change_id])

        start_time = self.connection.simulation.getTime()
        dur = self.connection.trafficlight.getPhaseDuration(next(iter(phase_changes.keys())))
        done = False
        while not done and self.connection.simulation.getTime() - start_time < dur - 0.1:
            done, my_cars, my_time = self._simulate_step()
            arrived_cars |= my_cars
            accumulated_travel_time += my_time

        return arrived_cars, accumulated_travel_time

    # def _print_event_q(self):
    #     s = ''
    #     for event in self.events:
    #         if event.event_type == EventType.ASK_FOR_ACTION:
    #             s += "ask "
    #         elif event.event_type == EventType.SWITCH_TO_GREEN:
    #             s += "to_green "
    #         elif event.event_type == EventType.SWITCH_TO_RED:
    #             s += "to_red "
    #
    #         s += str(event.simulation_time) + " " + str(event.jun_intern_id) + "; "
    #
    #     print(s)

    def _take_action(self, action):

        for i, act in enumerate(action):
            if act is not None:
                if self.curr_phases[i] == act:
                    self.events.append(Event(i, EventType.ASK_FOR_ACTION,
                                             self.connection.simulation.getTime() + self.green_dur))
                else:
                    self.curr_phases[i] = act
                    self.connection.trafficlight.setPhase(self.junctions[i],
                                                          self.connection.trafficlight.getPhase(self.junctions[i]) + 1)
                    self.events.append(
                        Event(i, EventType.SWITCH_TO_RED, self.connection.simulation.getTime() + self.yellow_dur))

        self.events = sorted(self.events, key=lambda evt: evt.simulation_time)

        arrived_cars = set()

        accumulated_travel_time = 0

        rewards = {}
        pressures = {}

        TIME_EPS = 0.1
        next_break = self.events[0].simulation_time
        event_it = 0

        for event in self.events:
            if event.simulation_time - next_break < TIME_EPS:
                next_break = event.simulation_time
                event_it += 1

        done = False
        while not done and next_break - self.connection.simulation.getTime() > TIME_EPS:
            done, my_cars, my_time = self._simulate_step()
            arrived_cars |= my_cars
            accumulated_travel_time += my_time

        for i in range(event_it):
            if self.events[i].event_type == EventType.ASK_FOR_ACTION:
                self.ret_state[self.events[i].jun_intern_id] = True
            elif self.events[i].event_type == EventType.SWITCH_TO_RED:
                self.connection.trafficlight.setPhase(self.junctions[self.events[i].jun_intern_id],
                                                      self.connection.trafficlight.getPhase(
                                                          self.junctions[self.events[i].jun_intern_id]) + 1)
                self.events.append(
                    Event(self.events[i].jun_intern_id, EventType.SWITCH_TO_GREEN,
                          self.connection.simulation.getTime() + self.red_dur))
            else:
                self.connection.trafficlight.setPhase(self.junctions[self.events[i].jun_intern_id],
                                                          3 * self.curr_phases[self.events[i].jun_intern_id])
                self.events.append(
                    Event(self.events[i].jun_intern_id, EventType.ASK_FOR_ACTION,
                          self.connection.simulation.getTime() + self.green_dur))

        self.events = self.events[event_it:]

        for tls_id in self.junctions:

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

            pressures[tls_id] = pressure
            rewards[tls_id] = -pressure

        self.throughput += len(arrived_cars)
        self.travel_time += accumulated_travel_time

        return done, np.mean(list(rewards.values())), {
            'reward': self.dict_states_to_array(rewards),
            'pressure': pressures,
            'travel_time': self.get_travel_time(),
            'throughput': self.get_throughput()
        }

    def _reset(self):
        self.travel_time = 0
        self.throughput = 0
        self.previous_actions = {}
        for junction in self.junctions:
            cluster = self.cluster_map.get(junction)
            if cluster:
                for jun, _ in cluster["tls_to_phases"].items():
                    self.previous_actions[jun] = (0, 1, 2, 3)
            else:
                self.previous_actions[junction] = (0, 1, 2, 3)
        self.traveling_cars = {}

        self.curr_phases = [-1] * len(self.junctions)

        self.events = []
        self.ret_state = [True] * len(self.junctions)

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
            self.clusters,
            self.max_steps,
            self.env_name
        )

    def __init__(self, sumocfg_file_path, vehicle_generator_config,
                 max_steps=1500, env_name=None,
                 junctions=None,
                 traffic_movements=TRAFFIC_MOVEMENTS,
                 traffic_lights_phases=TRAFFICLIGHTS_PHASES,
                 light_duration=LIGHT_DURATION,
                 clusters=None):
        super().__init__(sumocfg_file_path, vehicle_generator_config, max_steps, env_name=env_name)
        if junctions is None:
            junctions = ['gneJ18']
        self.junctions = junctions
        self.traffic_movements = traffic_movements
        self.traffic_lights_phases = traffic_lights_phases
        self.light_duration = light_duration
        self.clusters = clusters
