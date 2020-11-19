from collections import namedtuple

MAX_DISTANCE = 1333333333337

Edge = namedtuple('Edge', 'u v name')

class Node:
    def __init__(self, id, is_dead_end):
        self.id = id
        self.is_dead_end = is_dead_end
        self.distance = MAX_DISTANCE
        self.parent_edges = []
        self.edges = []

    def add_edge(self, v, name):
        self.edges.append(Edge(self, v, name))


def _reconstruct_and_get_routes(start, dead_ends):
    routes = []
    for dead_end in dead_ends:
        if dead_end != start:

            edges = []
            current = dead_end
            edges.append([current.parent_edges[0]])
            run = 1

            while run:
                new_edges = []
                for edge_list in edges:
                    last = edge_list[-1]

                    if last.u == start:
                        run = 0
                        break

                    for parent_edge in last.u.parent_edges:
                        tmp = edge_list.copy()
                        tmp.append(parent_edge)
                        new_edges.append(tmp)

                if run:
                    edges = new_edges

            for edge_list in edges:
                route = ''
                for edge in edge_list:
                    route = edge.name + ' ' + route

                routes.append(route[:-1])

    return routes


def generate_routes(graph, filename):
    routes = []
    dead_ends = list(filter(lambda node: node.is_dead_end, graph))

    for dead_end in dead_ends:

        for u in graph:
            u.parent_edges = []
            u.distance = MAX_DISTANCE

        dead_end.distance = 0
        stack = [dead_end]

        for u in stack:
            for edge in u.edges:
                v = edge.v
                if v.distance >= u.distance + 1 and len(v.parent_edges) < 3:
                    v.parent_edges.append(edge)
                    v.distance = u.distance + 1
                    stack.append(v)

        routes.extend(_reconstruct_and_get_routes(dead_end, dead_ends))

    with open(filename, 'w') as file:
        file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        file.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n')

        for i, route in enumerate(routes):
            file.write('\t<route edges="%s" id="route_%d"/>\n' % (route, i))

        file.write('</routes>')

