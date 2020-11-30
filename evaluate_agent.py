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
    evaluator = Evaluator.from_file("jsons/evaluators/osm_test.json")

    osm_phase_map = {
        0: ["gneE8_0", "gneE8_1", "166869096#1_0", "166869096#1_1",
            "166869096#1_2", "166869096#1_3", "gneE16_0", "gneE16_1"],
        1: ["gneE7_0", "gneE7_1", "gneE18_0", "gneE18_1"],
        2: ["gneE8_0", "gneE8_1", "gneE8_2", "gneE8_3", "gneE16_0", "gneE16_1"],
        3: ["166869096#1_0", "166869096#1_1", "gneE1_0", "gneE1_1", "gneE18_0", "gneE18_1"]
    }

    controller_map_osm = {
        "cluster1": VehicleNumberController(osm_phase_map)
    }

    all_metrics = evaluator.evaluate_all_dicts([controller_map_osm])

    for i, set_metrics in enumerate(all_metrics):
        print('SET %d' % i)
        for test_name, metric_dict in set_metrics.items():
            print(test_name.upper(), end=' ')
            for metric_name, val in metric_dict.items():
                print('%s=%d' % (metric_name, val), end=' ')
            print()


if __name__ == '__main__':
    main()

