from pathlib import Path

from evaluation.evaluator import Evaluator
from models.frap import Frap
from settings import PROJECT_ROOT, JSONS_FOLDER
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
    # evaluator = Evaluator.from_file("jsons/evaluators/2v2_small_subset.json")
    evaluator = Evaluator.from_file("jsons/evaluators/4v4_eq_vert_hori.json")

    osm_phase_map = {
        0: ["gneE8_0", "gneE8_1", "166869096#1_0", "166869096#1_1",
            "166869096#1_2", "166869096#1_3", "gneE16_0", "gneE16_1"],
        1: ["gneE7_0", "gneE7_1", "gneE18_0", "gneE18_1"],
        2: ["gneE8_0", "gneE8_1", "gneE8_2", "gneE8_3", "gneE16_0", "gneE16_1"],
        3: ["166869096#1_0", "166869096#1_1", "gneE1_0", "gneE1_1", "gneE18_0", "gneE18_1"]
    }

    controller1 = VehicleNumberController(osm_phase_map)

    phase_map = get_phase_map()
    controllers_2v2 = [VehicleNumberController(phase_map[tls_id]) for tls_id in phase_map.keys()]

    controller2 = TimedCyclicSwitchController(list(range(8)), [5] * 8)

    controller_rand = RandomSwitchController(list(range(8)))

    training_state = TrainingState.from_path(
        Path('saved', 'aaai-multi', 'frap', '4v4', 'frap_2020-12-13.16-24-11-166272', 'states',
             'ep_246_frap_2020-12-13.16-24-11-166272.tar'))
    model_4v4 = training_state.model
    model_4v4 = model_4v4.eval()

    model_4v4_controller = ModelController(model_4v4)

    phase_4v4 = get_phase_map_four_v_four()

    vehicle_num_controller_4v4 = [VehicleNumberController(phase_4v4[tls_id]) for tls_id in phase_4v4.keys()]

    metrics = evaluator.evaluate_traffic_controllers(
        [controller_rand, vehicle_num_controller_4v4, model_4v4_controller])

    for env_n in metrics[0].keys():
        print(env_n)
        for m in metrics:
            print(m[env_n])


if __name__ == '__main__':
    main()
