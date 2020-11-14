x = 0

basic_phases = [
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

def _get_phase_map(phases, lanes_names):
    phase_map = {}
    for phase in range(len(phases)):
        incoming = []
        for i in range(5):
            for j in range(5):
                if phases[phase][i][j]:
                    incoming.append(lanes_names[i][j])

        phase_map[phase] = incoming

    return phase_map