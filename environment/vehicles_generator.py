from abc import ABC
from abc import abstractmethod
from collections import namedtuple
import math
import random
import traci
import traci.constants as tc
import re

SinParameters = namedtuple("SinParameters", ["amplitude", "multiplier", "start", "min"])

SIN_ARG_DIVIDER = 10000000


class Lane:
    def __init__(self, lane_id):
        super().__init__()

        edge_id = traci.lane.getEdgeID(lane_id)
        # lane index from laneID
        self.lane_id = lane_id
        self.index = int(re.sub(f"{edge_id}_", "", lane_id))

        self.route_id = None

        start_edge = traci.lane.getEdgeID(lane_id)
        links = traci.lane.getLinks(lane_id)

        for link in links:
            end_edge = traci.lane.getEdgeID(link[0])
            for route in traci.route.getIDList():
                route_edges = traci.route.getEdges(route)
                if start_edge in route_edges and end_edge in route_edges:
                    self.route_id = route

        assert self.route_id is not None, f"unable to find route for {self.lane_id}"

        self.car_ids = 0

    def add_car(self):
        car_id = f"{self.lane_id}_{self.car_ids}"
        self.car_ids += 1
        traci.vehicle.add(
            vehID=car_id,
            routeID=self.route_id,
            departLane=self.index,
            departSpeed="max",
        )

    def reset_spawning_data(self):
        self.car_ids = 0

    def __str__(self):
        return f"lane: {self.lane_id} route: {self.route_id}"


class VehiclesGenerator(ABC):
    def __init__(self):
        self.lanes = {}
        self.last_spawns = {}
        self.lanes_periods = {}

    @abstractmethod
    def add_lane(self, lane):
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


class XMLGenerator(VehiclesGenerator):
    def __init__(self):
        super().__init__()

    def add_lane(self, lane, active):
        if not active:
            return

        spawn_period = traci.lane.getParameter(lane, "period")

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


class ConstGenerator(VehiclesGenerator):
    def __init__(self):
        super().__init__()

    def add_lane(self, lane, active, period):
        if not active:
            return

        self.lanes[lane] = Lane(lane)
        self.lanes_periods[lane] = period
        self.last_spawns[lane] = 0

    def generate_vehicles(self, time):
        for lane_id, lane in self.lanes.items():
            self._create_vehicle(lane_id, lane, time)


class SinusoidalGenerator(VehiclesGenerator):
    def __init__(self):
        super().__init__()
        self.last_time = 0
        self.time_sum = 0

        self.sin_parameters = {}

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


class WidgetGenerator(VehiclesGenerator):
    def __init__(self):
        super().__init__()

    def add_lane(self, lane, active, period):
        if not active:
            return

        self.lanes[lane] = Lane(lane)
        self.lanes_periods[lane] = period
        self.last_spawns[lane] = 0

    def generate_vehicles(self, time):
        for lane_id, lane in self.lanes.items():
            self._create_vehicle(lane_id, lane, time)

    def get_periods(self):
        return self.lanes_periods

    def set_periods(self, periods):
        for key in periods:
            self.lanes_periods[key] = periods[key]


class RandomGenerator(VehiclesGenerator):
    def __init__(self):
        super().__init__()
        self.arguments = {}

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
