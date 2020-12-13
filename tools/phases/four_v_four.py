from tools.phases.common import _get_phase_map, basic_phases

x = 0

def _lanes_name_mapper(pre):
    lanes_names_r0 = [x, pre[0][1] + '_0', pre[0][1] + '_1', pre[0][1] + '_2', x]
    lanes_names_r1 = [pre[1][0] + '_2', x, x, x, pre[1][2] + '_0']
    lanes_names_r2 = [pre[1][0] + '_1', x, x, x, pre[1][2] + '_1']
    lanes_names_r3 = [pre[1][0] + '_0', x, x, x, pre[1][2] + '_2']
    lanes_names_r4 = [x, pre[2][1] + '_2', pre[2][1] + '_1', pre[2][1] + '_0', x]

    return [lanes_names_r0, lanes_names_r1, lanes_names_r2, lanes_names_r3, lanes_names_r4]

gneJ23_pre_lanes_names = [
    [x,             'gneE20',   x],
    ['-gneE19',     x,          'gneE18'],
    [x,             '-gneE21',  x]
]

gneJ22_pre_lanes_names = [
    [x,             'gneE25',   x],
    ['-gneE18',     x,          'gneE17'],
    [x,             '-gneE26',  x]
]

gneJ21_pre_lanes_names = [
    [x,             'gneE30',   x],
    ['-gneE17',     x,          'gneE16'],
    [x,             '-gneE31',  x]
]

gneJ20_pre_lanes_names = [
    [x,             'gneE35',   x],
    ['-gneE16',     x,          'gneE15'],
    [x,             '-gneE36',  x]
]

gneJ17_pre_lanes_names = [
    [x,             'gneE36',   x],
    ['gneE13',     x,          '-gneE14'],
    [x,             '-gneE37',  x]
]

gneJ16_pre_lanes_names = [
    [x,             'gneE31',   x],
    ['gneE12',     x,          '-gneE13'],
    [x,             '-gneE32',  x]
]

gneJ15_pre_lanes_names = [
    [x,             'gneE26',   x],
    ['gneE11',     x,          '-gneE12'],
    [x,             '-gneE27',  x]
]

gneJ14_pre_lanes_names = [
    [x,             'gneE21',   x],
    ['gneE10',     x,          '-gneE11'],
    [x,             '-gneE22',  x]
]

gneJ7_pre_lanes_names = [
    [x,             'gneE22',   x],
    ['gneE5',     x,          '-gneE6'],
    [x,             '-gneE23',  x]
]

gneJ8_pre_lanes_names = [
    [x,             'gneE27',   x],
    ['gneE6',     x,          '-gneE7'],
    [x,             '-gneE28',  x]
]

gneJ9_pre_lanes_names = [
    [x,             'gneE32',   x],
    ['gneE7',     x,          '-gneE8'],
    [x,             '-gneE33',  x]
]

gneJ10_pre_lanes_names = [
    [x,             'gneE37',   x],
    ['gneE8',     x,          '-gneE9'],
    [x,             '-gneE38',  x]
]

gneJ4_pre_lanes_names = [
    [x,             'gneE38',   x],
    ['gneE3',     x,          '-gneE4'],
    [x,             '-gneE39',  x]
]

gneJ3_pre_lanes_names = [
    [x,             'gneE33',   x],
    ['gneE2',     x,          '-gneE3'],
    [x,             '-gneE34',  x]
]

gneJ2_pre_lanes_names = [
    [x,             'gneE28',   x],
    ['gneE1',     x,          '-gneE2'],
    [x,             '-gneE29',  x]
]

gneJ1_pre_lanes_names = [
    [x,             'gneE23',   x],
    ['gneE0',     x,          '-gneE1'],
    [x,             '-gneE24',  x]
]

def get_phase_map_four_v_four():
    return {'gneJ1': _get_phase_map(basic_phases, _lanes_name_mapper(gneJ1_pre_lanes_names)),
            'gneJ2': _get_phase_map(basic_phases, _lanes_name_mapper(gneJ2_pre_lanes_names)),
            'gneJ3': _get_phase_map(basic_phases, _lanes_name_mapper(gneJ3_pre_lanes_names)),
            'gneJ4': _get_phase_map(basic_phases, _lanes_name_mapper(gneJ4_pre_lanes_names)),
            'gneJ7': _get_phase_map(basic_phases, _lanes_name_mapper(gneJ7_pre_lanes_names)),
            'gneJ8': _get_phase_map(basic_phases, _lanes_name_mapper(gneJ8_pre_lanes_names)),
            'gneJ9': _get_phase_map(basic_phases, _lanes_name_mapper(gneJ9_pre_lanes_names)),
            'gneJ10': _get_phase_map(basic_phases, _lanes_name_mapper(gneJ10_pre_lanes_names)),
            'gneJ14': _get_phase_map(basic_phases, _lanes_name_mapper(gneJ14_pre_lanes_names)),
            'gneJ15': _get_phase_map(basic_phases, _lanes_name_mapper(gneJ15_pre_lanes_names)),
            'gneJ16': _get_phase_map(basic_phases, _lanes_name_mapper(gneJ16_pre_lanes_names)),
            'gneJ17': _get_phase_map(basic_phases, _lanes_name_mapper(gneJ17_pre_lanes_names)),
            'gneJ20': _get_phase_map(basic_phases, _lanes_name_mapper(gneJ20_pre_lanes_names)),
            'gneJ21': _get_phase_map(basic_phases, _lanes_name_mapper(gneJ21_pre_lanes_names)),
            'gneJ22': _get_phase_map(basic_phases, _lanes_name_mapper(gneJ22_pre_lanes_names)),
            'gneJ23': _get_phase_map(basic_phases, _lanes_name_mapper(gneJ23_pre_lanes_names))}

