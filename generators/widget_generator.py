from generators.lane import Lane
from generators.vehicles_generator import VehiclesGenerator


class WidgetGenerator(VehiclesGenerator):
    def __init__(self, connection, lanes):
        self.active_lanes = {}
        super().__init__(connection, lanes)

    def add_lane(self, lane, active, period):
        self.active_lanes[lane] = active

        self.lanes[lane] = Lane(self.connection, lane)
        self.lanes_periods[lane] = period
        self.last_spawns[lane] = 0

    def generate_vehicles(self, time):
        for lane_id, lane in self.lanes.items():
            if self.active_lanes[lane_id]:
                self._create_vehicle(lane_id, lane, time)

    def update(self, lanes_periods, active_lanes):
        self.lanes_periods = lanes_periods
        self.active_lanes = active_lanes

    def get_periods(self):
        return self.lanes_periods

    def get_active_lanes(self):
        return self.active_lanes
