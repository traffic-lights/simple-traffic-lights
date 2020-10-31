from pathlib import Path

import torch

from evaluation.evaluator import Evaluator
from traffic_controllers.cyclic_switch_controllers import TimedCyclicSwitchController
from traffic_controllers.model_controller import ModelController
from traffic_controllers.vehicle_number_controller import VehicleNumberController
from phases import *

from trainings.training_parameters import TrainingState

controller1 = VehicleNumberController(get_phase_map())
controller2 = TimedCyclicSwitchController(range(8), [5] * 8)

training_state = TrainingState.from_path(
    Path('saved', 'aaai-random', 'frap', 'frap_2020-10-12.21-30-17-802491', 'states',
         'ep_6_frap_2020-10-12.21-30-17-802491.tar'))

model_controller = ModelController(training_state.model)

evaluator = Evaluator.from_file("test_framework/configs/test.json")

metrics = evaluator.evaluate_traffic_controllers([model_controller, controller1, controller2])

for m in metrics:
    print(m)
