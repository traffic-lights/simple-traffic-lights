import re
import traci

class Lane:
    def __init__(self, lane_id):

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
