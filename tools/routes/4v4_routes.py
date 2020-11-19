from tools.routes.generate_route_file import Node, generate_routes

def symb_to_name(symbol):
    type = symbol[0]
    symbol = symbol[1:]
    if symbol[0] == '-':
        return '-' + 'gne' + type.upper() + symbol[1:]
    return 'gne' + type.upper() + symbol

syst = [
    [0, 0, 'j26', 0, 'j28', 0, 'j30', 0, 'j32', 0, 0],
    [0, 0, 'e20', 0, 'e25', 0, 'e30', 0, 'e35', 0, 0],
    ['j24', 'e-19', 'j23', 'e-18', 'j22', 'e-17', 'j21', 'e-16', 'j20', 'e-15', 'j15'],
    [0, 0, 'e21', 0, 'e26', 0, 'e31', 0, 'e36', 0, 0],
    ['j13', 'e10', 'j14', 'e11', 'j15', 'e12', 'j16', 'e13', 'j17', 'e14', 'j18'],
    [0, 0, 'e22', 0, 'e27', 0, 'e32', 0, 'e37', 0, 0],
    ['j6', 'e5', 'j7', 'e6', 'j8', 'e7', 'j9', 'e8', 'j10', 'e9', 'j11'],
    [0, 0, 'e23', 0, 'e28', 0, 'e33', 0, 'e38', 0, 0],
    ['j0', 'e0', 'j1', 'e1', 'j2', 'e2', 'j3', 'e3', 'j4', 'e4', 'j5'],
    [0, 0, 'e24', 0, 'e29', 0, 'e34', 0, 'e39', 0, 0],
    [0, 0, 'j27', 0, 'j29', 0, 'j31', 0, 'j33', 0, 0],
]

def graph_from_syst(syst, dim):
    graph = []
    for _ in range(dim):
        graph.append([0]*dim)
    cnt = 0
    for i in range(dim):
        for j in range(dim):
            if syst[i][j] and syst[i][j][0] == 'j':
                graph[i][j] = Node(cnt, i == 0 or i == dim-1 or j == 0 or j == dim-1)
                cnt += 1

    for i in range(dim):
        for j in range(dim):
            if syst[i][j] and syst[i][j][0] == 'e':
                name = symb_to_name(syst[i][j])
                neg_name = '-'+name if name[0] != '-' else name[1:]
                if syst[i-1][j]: # vertical
                    graph[i-1][j].add_edge(graph[i+1][j], name)
                    graph[i+1][j].add_edge(graph[i-1][j], neg_name)
                else: # horizontal
                    graph[i][j-1].add_edge(graph[i][j+1], name)
                    graph[i][j+1].add_edge(graph[i][j-1], neg_name)

    graph_list = []
    print(graph[1][0])
    for i in range(dim):
        for j in range(dim):
            if graph[i][j]:
                graph_list.append(graph[i][j])

    return graph_list


if __name__ == '__main__':
    graph = graph_from_syst(syst, 11)

    for node in graph:
        print('is dead?: ', node.is_dead_end)
        for edge in node.edges:
            print('edge: ', edge.name)

    generate_routes(graph, 'routes.txt')