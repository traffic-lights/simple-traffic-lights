from generators.lane import Lane
from generators.vehicles_generator import VehiclesGenerator
import connection


class XMLGenerator(VehiclesGenerator):
    def add_lane(self, lane, active):
        if not active:
            return

        spawn_period = connection.lane.getParameter(lane, "period")

        try:
            spawn_period = int(spawn_period)
        except ValueError:
            if spawn_period != "None" and spawn_period != "":
                print(f"bad period value: {spawn_period}")

            return

        assert spawn_period >= 0, "negative spawn period"

        self.lanes[lane] = Lane(lane)
        self.lanes_periods[lane] = spawn_period
        self.last_spawns[lane] = 0

    def generate_vehicles(self, time):
        for lane_id, lane in self.lanes.items():
            self._create_vehicle(lane_id, lane, time)
