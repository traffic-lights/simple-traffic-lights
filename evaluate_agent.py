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
    evaluator = Evaluator.from_file("jsons/evaluators/2v2_all_equal.json")

    model = Frap()
    # model_path = str(Path(PROJECT_ROOT, 'model', 'params.pkl'))
    # model_w = torch.load(model_path, map_location='cpu')
    # model.load_state_dict(model_w['agent_state_dict']['model'])
    # model = model.eval()

    model_controller = ModelController(model)

    controller_map1 = {
        'gneJ25': model_controller,
        'gneJ26': model_controller,
        'gneJ27': model_controller,
        'gneJ28': model_controller,
    }


    tls_to_phase_map = get_phase_map()

    controller_map2 = {tls: VehicleNumberController(phase_map) for tls, phase_map in tls_to_phase_map.items()}
    controller_map3 = {tls: VehicleNumberPressureController(tls, phase_map) for tls, phase_map in
                       tls_to_phase_map.items()}
    #controller_map4 = {tls: RandomSwitchController(range(8)) for tls, phase_map in tls_to_phase_map.items()}

    all_metrics = evaluator.evaluate_all_dicts([controller_map2])

    for i, set_metrics in enumerate(all_metrics):
        print('SET %d' % i)
        for test_name, metric_dict in set_metrics.items():
            print(test_name.upper(), end=' ')
            for metric_name, val in metric_dict.items():
                print('%s=%d' % (metric_name, val), end=' ')
            print()


if __name__ == '__main__':
    main()

