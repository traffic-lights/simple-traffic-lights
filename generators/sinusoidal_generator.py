import math
from collections import namedtuple

from generators.lane import Lane
from generators.vehicles_generator import VehiclesGenerator

SinParameters = namedtuple("SinParameters", ["amplitude", "multiplier", "start", "min"])

SIN_ARG_DIVIDER = 10000000


class SinusoidalGenerator(VehiclesGenerator):
    def __init__(self, lanes):
        self.last_time = 0
        self.time_sum = 0

        self.sin_parameters = {}
        super().__init__(lanes)

    def add_lane(self, lane, active, amplitude, multiplier, start, min):
        if not active:
            return

        self.lanes[lane] = Lane(lane)
        self.sin_parameters[lane] = SinParameters(amplitude / 2, multiplier, start, min)
        self.lanes_periods[lane] = self._calcualate_period(lane)
        self.last_spawns[lane] = 0

    def generate_vehicles(self, time):
        self._update(time)

        for lane_id, lane in self.lanes.items():
            if self._create_vehicle(lane_id, lane, time):
                self.lanes_periods[lane_id] = self._calcualate_period(lane_id)

    def _calcualate_period(self, lane_id):
        params = self.sin_parameters[lane_id]
        arg = (self.time_sum / SIN_ARG_DIVIDER) * math.pi + params.start * math.pi
        sin_value = math.sin(params.multiplier * (arg))
        return params.amplitude * sin_value + params.amplitude + params.min

    def _update(self, time):
        dt = time - self.last_time
        if dt < 0:
            dt = time

        if time != self.last_time:
            self.time_sum += dt

        self.last_time = time
