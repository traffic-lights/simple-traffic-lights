from abc import ABC
from abc import abstractmethod


class VehiclesGenerator(ABC):
    def __init__(self, connection, lanes):
        self.connection = connection
        self.lanes = {}
        self.last_spawns = {}
        self.lanes_periods = {}

        for lane in lanes:
            self.add_lane(**lane)

    @staticmethod
    def from_config_dict(connection, config_dict: dict):
        """
        Creates VehicleGenerator with type and lanes specified in config
        :param connection:
        :param config_dict: {'type': 'const', 'lanes': [{''}]}
        :return:
        """
        from generators import GENERATORS_TYPE_MAPPER

        return GENERATORS_TYPE_MAPPER[config_dict['type']](connection=connection, lanes=config_dict['lanes'])

    @abstractmethod
    def add_lane(self, *args, **kwargs):
        pass

    @abstractmethod
    def generate_vehicles(self, time):
        pass

    def _create_vehicle(self, lane_id, lane, time):
        dt = time - self.last_spawns[lane_id]
        if (dt >= self.lanes_periods[lane_id]) or self.last_spawns[lane_id] == 0:
            # print(f"{time - self.last_spawns[lane_id]} {self.lanes_periods[lane_id]}")
            lane.add_car()
            self.last_spawns[lane_id] = time
            return True

        return False

    def reset(self):
        for lane in self.lanes.values():
            self.last_spawns[lane.lane_id] = 0
            lane.reset_spawning_data()
