import math

# phase_to_incoming_lanes_map format:
# phase_to_incoming_lanes_map = {
#     phase1: [in1, in2, in3],
#     phase2: [in1, in4]
#     ...
#     eg.
#     2: ['gneE15_0', 'gneE15_1']
# }
from dataclasses import dataclass

from traffic_controllers.trafffic_controller import TrafficController


class VehicleNumberControllerRunner:
    def __init__(self, connection, phase_to_incoming_lanes_map):
        self.phase_to_incoming_lanes_map = phase_to_incoming_lanes_map
        self.connection = connection

    def __call__(self, state):
        vehicle_per_lane = []
        for phase, incoming_lanes in self.phase_to_incoming_lanes_map.items():
            cnt = 0
            for lane in incoming_lanes:
                cnt += self.connection.lane.getLastStepVehicleNumber(lane)

            vehicle_per_lane.append((phase, cnt))

        return sorted(vehicle_per_lane, key=lambda x: x[1], reverse=True)[0][0]


@dataclass
class VehicleNumberController(TrafficController):
    phase_to_incoming_lanes_map: dict

    def with_connection(self, connection):
        return VehicleNumberControllerRunner(connection, self.phase_to_incoming_lanes_map)


class VehicleNumberPhaseDurationControllerRunner:
    def __init__(self, connection, min_duration, max_duration, phase_to_incoming_lanes_map):
        self.connection = connection
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.phase_to_incoming_lanes_map = phase_to_incoming_lanes_map

        self.phases = list(phase_to_incoming_lanes_map.keys())
        self.iter = len(self.phases) - 1
        self.curr_phase_end_time = 0

    def _calculate_curr_phase_end_time(self):
        total = 0
        lanes = set()
        vehicle_per_lane = []
        for phase, incoming_lanes in self.phase_to_incoming_lanes_map.items():
            cnt = 0
            for lane in incoming_lanes:
                vehicles_on_lane = self.connection.lane.getLastStepVehicleNumber(lane)
                cnt += vehicles_on_lane
                if lane not in lanes:
                    lanes.add(lane)
                    total += vehicles_on_lane

            vehicle_per_lane.append((phase, cnt))

        if total:
            factor = math.sqrt(vehicle_per_lane[self.iter][1] / total)
        else:
            factor = 0

        self.curr_phase_end_time = self.connection.simulation.getTime() + self.min_duration + (
                self.max_duration - self.min_duration) * factor

    def __call__(self, state):
        if self.connection.simulation.getTime() >= self.curr_phase_end_time:
            self.iter = (self.iter + 1) % len(self.phases)
            self._calculate_curr_phase_end_time()

        return self.phases[self.iter]


@dataclass
class VehicleNumberPhaseDurationController(TrafficController):
    min_duration: float
    max_duration: float
    phase_to_incoming_lanes_map: dict

    def with_connection(self, connection):
        return VehicleNumberPhaseDurationControllerRunner(connection, self.min_duration, self.max_duration,
                                                          self.phase_to_incoming_lanes_map)


class VehicleNumberPressureControllerRunner:
    def __init__(self, connection, tls_id, phase_to_incoming_lanes_map):
        self.phase_to_incoming_lanes_map = phase_to_incoming_lanes_map
        self.connection = connection

        incomings = set()
        for ins in phase_to_incoming_lanes_map.values():
            incomings.update(ins)

        self.in_out_map = {}
        for entry in self.connection.trafficlight.getControlledLinks(tls_id):
            inc, out = entry[0]
            if inc in incomings:
                self.in_out_map[inc] = out

    def __call__(self, state):
        vehicle_per_lane = []
        for phase, incoming_lanes in self.phase_to_incoming_lanes_map.items():
            pressures = 0
            for lane in incoming_lanes:
                pressures += self.connection.lane.getLastStepVehicleNumber(
                    lane) - self.connection.lane.getLastStepVehicleNumber(
                    self.in_out_map[lane])

            vehicle_per_lane.append((phase, pressures))

        return sorted(vehicle_per_lane, key=lambda x: x[1], reverse=True)[0][0]


@dataclass
class VehicleNumberPressureController(TrafficController):
    tls_id: int
    phase_to_incoming_lanes_map: dict

    def with_connection(self, connection):
        return VehicleNumberPressureControllerRunner(connection, self.tls_id, self.phase_to_incoming_lanes_map)
