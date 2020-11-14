from tools.routes.generate_route_file import Node, generate_routes

j0 = Node(0, False)
j1 = Node(1, False)
j2 = Node(2, False)
j3 = Node(3, False)

de0 = Node(4, True)
de1 = Node(5, True)
de2 = Node(6, True)
de3 = Node(7, True)
de4 = Node(8, True)
de5 = Node(9, True)
de6 = Node(10, True)
de7 = Node(11, True)

j0.add_edge(de0, '-gneE27')
j0.add_edge(de7, 'gneE25')
j0.add_edge(j1, 'gneE21')
j0.add_edge(j2, '-gneE24')

j1.add_edge(j0, '-gneE21')
j1.add_edge(j3, 'gneE22')
j1.add_edge(de1, '-gneE28')
j1.add_edge(de2, '-gneE29')

j2.add_edge(j3, '-gneE23')
j2.add_edge(de5, '-gneE32')
j2.add_edge(de6, '-gneE33')
j2.add_edge(j0, 'gneE24')

j3.add_edge(j1, '-gneE22')
j3.add_edge(de3, '-gneE30')
j3.add_edge(de4, '-gneE31')
j3.add_edge(j2, 'gneE23')

de0.add_edge(j0, 'gneE27')
de1.add_edge(j1, 'gneE28')
de2.add_edge(j1, 'gneE29')
de3.add_edge(j3, 'gneE30')
de4.add_edge(j3, 'gneE31')
de5.add_edge(j2, 'gneE32')
de6.add_edge(j2, 'gneE33')
de7.add_edge(j0, '-gneE25')

graph = [j0, j1, j2, j3, de0, de1, de2, de3, de4, de5, de6, de7]


if __name__ == '__main__':
    generate_routes(graph, 'routes.txt')