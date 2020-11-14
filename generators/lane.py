import re
import numpy as np

class Lane:
    def __init__(self, connection, lane_id):
        self.connection = connection
        edge_id = self.connection.lane.getEdgeID(lane_id)
        # lane index from laneID
        self.lane_id = lane_id
        self.index = int(re.sub(f"{edge_id}_", "", lane_id))

        self.routes_ids = []

        start_edge = self.connection.lane.getEdgeID(lane_id)
        links = self.connection.lane.getLinks(lane_id)

        for link in links:
            # end_edge = self.connection.lane.getEdgeID(link[0])
            for route in self.connection.route.getIDList():
                route_edges = self.connection.route.getEdges(route)
                # if start_edge in route_edges and end_edge in route_edges:
                if start_edge in route_edges:
                    self.routes_ids.append(route)

        assert self.routes_ids, f"unable to find route for {self.lane_id}"

        # print(lane_id, self.routes_ids)

        self.car_ids = 0

    def add_car(self):
        car_id = f"{self.lane_id}_{self.car_ids}"
        self.car_ids += 1
        self.connection.vehicle.add(
            vehID=car_id,
            routeID=np.random.choice(self.routes_ids),
            departLane=self.index,
            departSpeed="max",
        )

    def reset_spawning_data(self):
        self.car_ids = 0

    def __str__(self):
        return f"lane: {self.lane_id} routes: {self.routes_ids}"
