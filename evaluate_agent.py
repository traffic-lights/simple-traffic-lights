from pathlib import Path

from evaluation.evaluator import Evaluator
from models.frap import Frap
from settings import PROJECT_ROOT
from tools.phases.two_v_two import get_phase_map
from traffic_controllers.cyclic_switch_controllers import TimedCyclicSwitchController, RandomSwitchController

import torch

from traffic_controllers.model_controller import ModelController
from traffic_controllers.vehicle_number_controller import VehicleNumberController, VehicleNumberPressureController

from trainings.training_parameters import TrainingState


def main():
    #evaluator = Evaluator.from_file("jsons/evaluators/osm_test.json")
    evaluator = Evaluator.from_file("jsons/evaluators/2v2_small_subset.json")

    osm_phase_map = {
        0: ["gneE8_0", "gneE8_1", "166869096#1_0", "166869096#1_1",
            "166869096#1_2", "166869096#1_3", "gneE16_0", "gneE16_1"],
        1: ["gneE7_0", "gneE7_1", "gneE18_0", "gneE18_1"],
        2: ["gneE8_0", "gneE8_1", "gneE8_2", "gneE8_3", "gneE16_0", "gneE16_1"],
        3: ["166869096#1_0", "166869096#1_1", "gneE1_0", "gneE1_1", "gneE18_0", "gneE18_1"]
    }

    controller1 = VehicleNumberController(osm_phase_map)

    controller2 = RandomSwitchController(range(8))

    phase_map = get_phase_map()
    print(phase_map.keys())
    controllers3 = [VehicleNumberController(phase) for tls_id, phase in phase_map.items()]

    training_state = TrainingState.from_path(
        Path('saved', 'aaai-multi', 'frap', 'frap_2020-11-30.22-37-20-340261', 'states',
             'ep_20_frap_2020-11-30.22-37-20-340261.tar'))
    model1 = training_state.model
    model1 = model1.eval()

    controller4 = ModelController(model1)

    metrics = evaluator.evaluate_traffic_controllers([controller4])

    for env_n in metrics[0].keys():
        print(env_n)
        for m in metrics:
            print(m[env_n])


if __name__ == '__main__':
    main()

