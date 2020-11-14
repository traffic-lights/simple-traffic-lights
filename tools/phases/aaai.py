from tools.phases.common import _get_phase_map, basic_phases

x = 0

aaai_lanes_names = [
    [x,          'gneE15_0',  'gneE15_1',   'gneE15_2',  x],
    ['gneE21_2',  x,              x,        x,          'gneE17_0'],
    ['gneE21_1',  x,              x,        x,          'gneE17_1'],
    ['gneE21_0',  x,              x,        x,          'gneE17_2'],
    [x,          'gneE19_2',  'gneE19_1',   'gneE19_0',  x]
]


def get_phase_map():
    return {'gneJ18': _get_phase_map(basic_phases, aaai_lanes_names)}