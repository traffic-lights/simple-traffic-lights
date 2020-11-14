from collections import namedtuple

Edge = namedtuple('Edge', 'u v name')

class Node:
    def __init__(self, id, is_dead_end):
        self.id = id
        self.is_dead_end = is_dead_end
        self.parent_edge = None
        self.edges = []

    def add_edge(self, v, name):
        self.edges.append(Edge(self, v, name))


def _reconstruct_and_get_routes(start, dead_ends):
    routes = []
    for dead_end in dead_ends:
        if dead_end != start:
            current = dead_end
            route = ''
            while current != start:
                route = current.parent_edge.name + ' ' + route
                current = current.parent_edge.u

            routes.append(route[:-1])

    return routes


def generate_routes(graph, filename):
    routes = []
    dead_ends = list(filter(lambda node: node.is_dead_end, graph))

    for dead_end in dead_ends:

        for u in graph:
            u.parent_edge = None

        dead_end.parent_edge = 1
        stack = [dead_end]

        for u in stack:
            for edge in u.edges:
                v = edge.v
                if not v.parent_edge:
                    v.parent_edge = edge
                    stack.append(v)

        routes.extend(_reconstruct_and_get_routes(dead_end, dead_ends))

    with open(filename, 'w') as file:
        file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        file.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n')

        for i, route in enumerate(routes):
            file.write('\t<route edges="%s" id="route_%d"/>\n' % (route, i))

        file.write('</routes>')
