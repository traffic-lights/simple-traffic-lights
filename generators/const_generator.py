from generators.lane import Lane
from generators.vehicles_generator import VehiclesGenerator


class ConstGenerator(VehiclesGenerator):
    def add_lane(self, lane, active, period):
        if not active:
            return

        self.lanes[lane] = Lane(lane)
        self.lanes_periods[lane] = period
        self.last_spawns[lane] = 0

    def generate_vehicles(self, time):
        for lane_id, lane in self.lanes.items():
            self._create_vehicle(lane_id, lane, time)