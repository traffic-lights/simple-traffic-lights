x = 0

phases = [
    [[x, 1, 0, 0, x],
     [0, x, x, x, 1],
     [1, x, x, x, 1],
     [1, x, x, x, 0],
     [x, 0, 0, 1, x]],

    [[x, 1, 0, 0, x],
     [0, x, x, x, 1],
     [0, x, x, x, 1],
     [1, x, x, x, 1],
     [x, 0, 0, 1, x]],

    [[x, 1, 0, 0, x],
     [1, x, x, x, 1],
     [1, x, x, x, 0],
     [1, x, x, x, 0],
     [x, 0, 0, 1, x]],

    [[x, 1, 0, 0, x],
     [1, x, x, x, 1],
     [0, x, x, x, 0],
     [1, x, x, x, 1],
     [x, 0, 0, 1, x]],

    [[x, 1, 1, 0, x],
     [0, x, x, x, 1],
     [0, x, x, x, 0],
     [1, x, x, x, 0],
     [x, 0, 1, 1, x]],

    [[x, 1, 0, 0, x],
     [0, x, x, x, 1],
     [0, x, x, x, 0],
     [1, x, x, x, 0],
     [x, 1, 1, 1, x]],

    [[x, 1, 1, 1, x],
     [0, x, x, x, 1],
     [0, x, x, x, 0],
     [1, x, x, x, 0],
     [x, 0, 0, 1, x]],

    [[x, 1, 0, 1, x],
     [0, x, x, x, 1],
     [0, x, x, x, 0],
     [1, x, x, x, 0],
     [x, 1, 0, 1, x]],
]


lanes_names = [
    [x,          'gneE15_0',  'gneE15_1',   'gneE15_2',  x],
    ['gneE21_2',  x,              x,        x,          'gneE17_0'],
    ['gneE21_1',  x,              x,        x,          'gneE17_1'],
    ['gneE21_0',  x,              x,        x,          'gneE17_2'],
    [x,          'gneE19_2',  'gneE19_1',   'gneE19_0',  x]
]


def get_phase_map():
    phase_map = {}
    for phase in range(len(phases)):
        incoming = []
        for i in range(5):
            for j in range(5):
                if phases[phase][i][j]:
                    incoming.append(lanes_names[i][j])

        phase_map[phase] = incoming

    return phase_map


