import traci
import math
import time

# phase_to_incoming_lanes_map format:
# phase_to_incoming_lanes_map = {
#     phase1: [in1, in2, in3],
#     phase2: [in1, in4]
#     ...
#     eg.
#     2: ['gneE15_0', 'gneE15_1']
# }


class VehicleNumberController:
    def __init__(self, phase_to_incoming_lanes_map):
        self.phase_to_incoming_lanes_map = phase_to_incoming_lanes_map

    def __call__(self, state):
        vehicle_per_lane = []
        for phase, incoming_lanes in self.phase_to_incoming_lanes_map.items():
            cnt = 0
            for lane in incoming_lanes:
                cnt += traci.lane.getLastStepVehicleNumber(lane)

            vehicle_per_lane.append((phase, cnt))

        return sorted(vehicle_per_lane, key=lambda x: x[1], reverse=True)[0][0]


class VehicleNumberPhaseDurationController:
    def __init__(self, min_duration, max_duration, time_f, phase_to_incoming_lanes_map):
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.time_f = time_f
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
                vehicles_on_lane = traci.lane.getLastStepVehicleNumber(lane)
                cnt += vehicles_on_lane
                if lane not in lanes:
                    lanes.add(lane)
                    total += vehicles_on_lane

            vehicle_per_lane.append((phase, cnt))

        if total:
            factor = math.sqrt(vehicle_per_lane[self.iter][1] / total)
        else:
            factor = 0

        print(factor)

        self.curr_phase_end_time = self.time_f() + self.min_duration + (self.max_duration - self.min_duration) * factor

    def __call__(self, state):
        if self.time_f() >= self.curr_phase_end_time:
            self.iter = (self.iter + 1) % len(self.phases)

        self._calculate_curr_phase_end_time()

        return self.phases[self.iter]


class BlockingVehicleNumberPhaseDurationController(VehicleNumberPhaseDurationController):
    def __call__(self, state):
        if self.time_f() < self.curr_phase_end_time:
            time.sleep(self.curr_phase_end_time - self.time_f())

        self.iter = (self.iter + 1) % len(self.phases)
        self._calculate_curr_phase_end_time()

        return self.phases[self.iter]

