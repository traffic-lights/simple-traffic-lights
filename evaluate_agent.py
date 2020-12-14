from pathlib import Path

from evaluation.evaluator import Evaluator
from models.frap import Frap
from settings import PROJECT_ROOT, JSONS_FOLDER
from tools.phases.aaai import get_phase_map_1v1
from tools.phases.two_v_two import get_phase_map
from tools.phases.four_v_four import get_phase_map_four_v_four
from traffic_controllers.cyclic_switch_controllers import TimedCyclicSwitchController, RandomSwitchController

import torch

from traffic_controllers.model_controller import ModelController
from traffic_controllers.vehicle_number_controller import VehicleNumberController, VehicleNumberPressureController, \
    VehicleNumberPressureControllerRunner

from trainings.training_parameters import TrainingState


def main():
    # evaluator = Evaluator.from_file("jsons/evaluators/example_test.json")
    # evaluator = Evaluator.from_file("jsons/evaluators/osm_test.json")
    evaluator = Evaluator.from_file("jsons/evaluators/2v2_small_subset.json")
    # evaluator = Evaluator.from_file('jsons/evaluators/4v4_eq_vert_hori.json')
    # evaluator = Evaluator.from_file('jsons/evaluators/example_test.json')

    osm_phase_map = {
        0: ["gneE8_0", "gneE8_1", "166869096#1_0", "166869096#1_1",
            "166869096#1_2", "166869096#1_3", "gneE16_0", "gneE16_1"],
        1: ["gneE7_0", "gneE7_1", "gneE18_0", "gneE18_1"],
        2: ["gneE8_0", "gneE8_1", "gneE8_2", "gneE8_3", "gneE16_0", "gneE16_1"],
        3: ["166869096#1_0", "166869096#1_1", "gneE1_0", "gneE1_1", "gneE18_0", "gneE18_1"]
    }

    controller1 = VehicleNumberController(osm_phase_map)

    phase_map = get_phase_map()
    vehicle_number_controller_2x2 = [VehicleNumberController(phase_map[tls_id]) for tls_id in phase_map.keys()]
    phase_map_1v1 = get_phase_map_1v1()
    vehicle_number_controller_1x1 = [VehicleNumberController(phase_map_1v1[tls_id]) for tls_id in phase_map_1v1.keys()]
    pressure_controller_1x1 = [VehicleNumberPressureController(tls_id, phase) for tls_id, phase in
                               phase_map_1v1.items()]

    timed_controller = TimedCyclicSwitchController(list(range(8)), [5] * 8)

    controller_rand = RandomSwitchController(list(range(8)))

    phase_map = get_phase_map()
    pressure_controller_2x2 = [VehicleNumberPressureController(tls_id, phase) for tls_id, phase in phase_map.items()]

    model_1v1_training_state = TrainingState.from_path(
        str(Path(PROJECT_ROOT, 'saved', 'aaai-random', 'frap', 'frap_2020-11-12.23-17-52-665056', 'states',
             'ep_9_frap_2020-11-12.23-17-52-665056.tar'))
    )
    model_1v1 = model_1v1_training_state.model
    model_1v1_controller = ModelController(model_1v1.eval())

    model_2v2_training_state = TrainingState.from_path(
        str(Path(PROJECT_ROOT, 'saved', 'aaai-multi', 'frap', 'frap_2020-12-13.12-43-58-417983', 'states',
             'ep_181_frap_2020-12-13.12-43-58-417983.tar'))
    )
    model_2v2 = model_2v2_training_state.model
    model_2v2_controller = ModelController(model_2v2.eval())

    # training_state = TrainingState.from_path(
    #     Path('ep_181_frap_2020-12-13.12-43-58-417983.tar'))
    # model1 = training_state.model
    # model1 = model1.eval()

    # controller_mod = ModelController(model1)

    controllers_2cyc_2model = [controller_rand,
                               timed_controller,
                               vehicle_number_controller_2x2,
                               pressure_controller_2x2]

    controllers_1x1 = [
        # controller_rand,
        # timed_controller,
        # vehicle_number_controller_1x1,
        # pressure_controller_1x1
        model_1v1_controller
    ]

    phase_map_4v4 = get_phase_map_four_v_four()
    controllers_4v4_vehicle = [VehicleNumberController(phase_map_4v4[tls_id]) for tls_id in phase_map_4v4.keys()]
    controllers_4v4_pressure = [VehicleNumberPressureController(tls_id, phase_map_4v4[tls_id]) for tls_id in
                                phase_map_4v4.keys()]

    # metrics = evaluator.evaluate_traffic_controllers(
    #     [controller_rand, controllers_4v4_vehicle, controllers_4v4_pressure], render=False)

    metrics = evaluator.evaluate_traffic_controllers([model_2v2_controller], render=False)

    for env_n in metrics[0].keys():
        print(env_n)
        for m in metrics:
            print(m[env_n])


if __name__ == '__main__':
    main()
