from abc import ABC
from abc import abstractmethod
from collections import namedtuple
import math
import traci
import traci.constants as tc
import re

SinParameters = namedtuple("SinParameters", ["amplitude", "multiplier", "start", "min"])

MAX_LANE_OCCUPANCY = 0.6
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

        self.next_timer = 0
        self.car_ids = 0

    def add_car(self, current_time, period):
        last_step_occupancy = traci.lane.getLastStepOccupancy(self.lane_id)
        if (
            self.next_timer - current_time <= 0
            and last_step_occupancy <= MAX_LANE_OCCUPANCY
        ):
            car_id = f"{self.lane_id}_{self.car_ids}"
            self.car_ids += 1
            traci.vehicle.add(
                vehID=car_id, routeID=self.route_id, departLane=self.index
            )
            self.next_timer = current_time + period

            return True
        return False

    def reset_spawning_data(self):
        self.next_timer = 0
        self.car_ids = 0

    def __str__(self):
        return f"lane: {self.lane_id} route: {self.route_id}"


class VehiclesGenerator(ABC):
    lanes = {}

    @classmethod
    @abstractmethod
    def add_lane(cls, lane):
        pass

    @classmethod
    @abstractmethod
    def generate_vehicles(cls, time):
        pass

    @classmethod
    @abstractmethod
    def _update(cls, time):
        pass

    @classmethod
    def reset(cls):
        for lane in cls.lanes.values():
            lane.reset_spawning_data()


class XMLGenerator(VehiclesGenerator):
    lanes_periods = {}

    @classmethod
    def add_lane(cls, lane, active):
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

        cls.lanes[lane] = Lane(lane)
        cls.lanes_periods[lane] = spawn_period

    @classmethod
    def generate_vehicles(cls, time):
        cls._update(time)

        for lane_id, lane in cls.lanes.items():
            period = cls.lanes_periods[lane_id]
            lane.add_car(time, period)
            # print(f"lane: {lane_id} period: {period}")

    @classmethod
    def _update(cls, time):
        pass


class ConstGenerator(VehiclesGenerator):
    lanes_periods = {}

    @classmethod
    def add_lane(cls, lane, active, period):
        if not active:
            return

        cls.lanes[lane] = Lane(lane)
        cls.lanes_periods[lane] = period

    @classmethod
    def generate_vehicles(cls, time):
        cls._update(time)

        for lane_id, lane in cls.lanes.items():
            period = cls.lanes_periods[lane_id]
            lane.add_car(time, period)
            # print(f"lane: {lane_id} period: {period}")

    @classmethod
    def _update(cls, time):
        pass


class SinusoidalGenerator(VehiclesGenerator):
    last_time = 0
    time_sum = 0

    sin_parameters = {}

    @classmethod
    def add_lane(cls, lane, active, amplitude, multiplier, start, min):
        if not active:
            return

        cls.lanes[lane] = Lane(lane)
        cls.sin_parameters[lane] = SinParameters(amplitude / 2, multiplier, start, min)

    @classmethod
    def generate_vehicles(cls, time):
        cls._update(time)

        for lane_id, lane in cls.lanes.items():
            params = cls.sin_parameters[lane_id]
            arg = (cls.time_sum / SIN_ARG_DIVIDER) * math.pi + params.start * math.pi
            sin_value = math.sin(params.multiplier * (arg))
            period = params.amplitude * sin_value + params.amplitude + params.min
            # print(f"period: {period}")
            lane.add_car(time, period)

    @classmethod
    def _update(cls, time):
        dt = time - cls.last_time
        if dt < 0:
            dt = time

        if time != cls.last_time:
            cls.time_sum += dt

        cls.last_time = time

        # print(f"time: {time} dt: {dt}")


class WidgetGenerator(VehiclesGenerator):
    lanes_periods = {}

    @classmethod
    def add_lane(cls, lane, active, period):
        if not active:
            return

        cls.lanes[lane] = Lane(lane)
        cls.lanes_periods[lane] = period

    @classmethod
    def generate_vehicles(cls, time):
        cls._update(time)

        for lane_id, lane in cls.lanes.items():
            period = cls.lanes_periods[lane_id]
            lane.add_car(time, period)
            # print(f"lane: {lane_id} period: {period}")

    @classmethod
    def _update(cls, time):
        pass

    @classmethod
    def get_periods(cls):
        return cls.lanes_periods

    @classmethod
    def set_periods(cls, periods):
        for key in periods:
            cls.lanes_periods[key] = periods[key]
