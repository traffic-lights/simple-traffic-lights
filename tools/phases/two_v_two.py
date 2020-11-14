from tools.phases.common import _get_phase_map, basic_phases

x = 0

gneJ25_lanes_names = [
    [x,          'gneE27_0',  'gneE27_1',   'gneE27_2',  x],
    ['-gneE25_2',  x,              x,        x,          '-gneE21_0'],
    ['-gneE25_1',  x,              x,        x,          '-gneE21_1'],
    ['-gneE25_0',  x,              x,        x,          '-gneE21_2'],
    [x,          'gneE24_2',  'gneE24_1',   'gneE24_0',  x]
]

gneJ26_lanes_names = [
    [x,          'gneE28_0',  'gneE28_1',   'gneE28_2',  x],
    ['gneE21_2',  x,              x,        x,          'gneE29_0'],
    ['gneE21_1',  x,              x,        x,          'gneE29_1'],
    ['gneE21_0',  x,              x,        x,          'gneE29_2'],
    [x,          '-gneE22_2',  '-gneE22_1',   '-gneE22_0',  x]
]

gneJ27_lanes_names = [
    [x,          'gneE22_0',  'gneE22_1',   'gneE22_2',  x],
    ['-gneE23_2',  x,              x,        x,          'gneE30_0'],
    ['-gneE23_1',  x,              x,        x,          'gneE30_1'],
    ['-gneE23_0',  x,              x,        x,          'gneE30_2'],
    [x,          'gneE31_2',  'gneE31_1',   'gneE31_0',  x]
]

gneJ28_lanes_names = [
    [x,          '-gneE24_0',  '-gneE24_1',   '-gneE24_2',  x],
    ['gneE33_2',  x,              x,        x,          'gneE23_0'],
    ['gneE33_1',  x,              x,        x,          'gneE23_1'],
    ['gneE33_0',  x,              x,        x,          'gneE23_2'],
    [x,          'gneE32_2',  'gneE32_1',   'gneE32_0',  x]
]

def get_phase_map():
    return {'gneJ25': _get_phase_map(basic_phases, gneJ25_lanes_names),
            'gneJ26': _get_phase_map(basic_phases, gneJ26_lanes_names),
            'gneJ27': _get_phase_map(basic_phases, gneJ27_lanes_names),
            'gneJ28': _get_phase_map(basic_phases, gneJ28_lanes_names)}