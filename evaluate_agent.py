from pathlib import Path

from evaluation.evaluator import Evaluator
from tools.phases.two_v_two import get_phase_map
from traffic_controllers.cyclic_switch_controllers import TimedCyclicSwitchController

import torch

from traffic_controllers.model_controller import ModelController
from traffic_controllers.vehicle_number_controller import VehicleNumberController

from trainings.training_parameters import TrainingState

def main():
    evaluator = Evaluator.from_file("test_framework/configs/test_2v2.json")

    cyclic_controller = TimedCyclicSwitchController(list(range(8)), [3] * 8)

    training_state = TrainingState.from_path(
        Path('trainings', 'saved', 'aaai', 'frap', 'frap_2020-11-12.11-56-42-474684', 'states',
             'ep_6_frap_2020-11-12.11-56-42-474684.tar'))

    model_controller = ModelController(training_state.model)

    controller_map1 = {
        'gneJ25': model_controller,
        'gneJ26': model_controller,
        'gneJ27': model_controller,
        'gneJ28': model_controller,
    }

    tls_to_phase_map = get_phase_map()

    controller_map2 = {tls: VehicleNumberController(phase_map) for tls, phase_map in tls_to_phase_map.items()}

    controller_map3 = controller_map2.copy()

    controller_map3['gneJ25'] = cyclic_controller
    controller_map3['gneJ26'] = cyclic_controller

    all_metrics = evaluator.evaluate_all_dicts([controller_map1, controller_map2, controller_map3])

    for metrics in all_metrics:
        print(metrics)


if __name__ == '__main__':
    main()
