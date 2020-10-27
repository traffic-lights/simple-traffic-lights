import random

from generators.lane import Lane
from generators.vehicles_generator import VehiclesGenerator


class RandomGenerator(VehiclesGenerator):
    def __init__(self, lanes):
        self.arguments = {}
        super().__init__(lanes)

    def add_lane(self, lane, active, min_period, max_period):
        if not active:
            return
        self.lanes[lane] = Lane(lane)
        self.arguments[lane] = (min_period, max_period)
        self.lanes_periods[lane] = random.uniform(min_period, max_period)
        self.last_spawns[lane] = 0

    def generate_vehicles(self, time):
        for lane_id, lane in self.lanes.items():
            if self._create_vehicle(lane_id, lane, time):
                a, b = self.arguments[lane_id]
                self.lanes_periods[lane_id] = random.uniform(a, b)
