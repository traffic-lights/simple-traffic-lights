from pathlib import Path
from trafffic_controller import TrafficController

from FixedControllers.cyclic_switch_controllers import TimedCyclicSwitchController
from FixedControllers.vehicle_number_controller import VehicleNumberController
from phases import *

from evaluation.evaluator import Evaluator

controller1 = TrafficController(VehicleNumberController(get_phase_map()))
controller2 = TrafficController(TimedCyclicSwitchController(range(8), [5]*8))

evaluator = Evaluator.from_file("test_framework/configs/test.json")

metrics = evaluator.evaluate_traffic_controllers([controller1, controller2])

print(metrics)