from pathlib import Path

import torch

from environments.sumo_env import SumoEnv
from evaluation.evaluator import Evaluator, evaluate_controller
from models.frap import Frap
from settings import PROJECT_ROOT, JSONS_FOLDER
from traffic_controllers.cyclic_switch_controllers import TimedCyclicSwitchController
from traffic_controllers.model_controller import ModelController
from traffic_controllers.vehicle_number_controller import VehicleNumberController
from phases import *

from trainings.training_parameters import TrainingState

controller1 = VehicleNumberController(get_phase_map())
controller2 = TimedCyclicSwitchController(range(8), [5] * 8)

training_state = TrainingState.from_path(
    Path('saved', 'aaai-random', 'frap', 'frap_2020-10-12.21-30-17-802491', 'states',
         'ep_7_frap_2020-10-12.21-30-17-802491.tar'))

evaluator = Evaluator.from_file(Path(JSONS_FOLDER, 'evaluators', 'example_test.json'))

model1 = training_state.model
model1 = model1.eval()

model2 = Frap()
model_path = str(Path(PROJECT_ROOT, 'saved', 'rlpyt', 'params-7.pkl'))
model_w = torch.load(model_path, map_location='cpu')
model2.load_state_dict(model_w['agent_state_dict']['model'])
model2 = model2.eval()

model3 = Frap()
model_path = str(Path(PROJECT_ROOT, 'saved', 'rlpyt', 'params-6.pkl'))
model_w = torch.load(model_path, map_location='cpu')
model3.load_state_dict(model_w['agent_state_dict']['model'])
model3 = model3.eval()

model1_controller = ModelController(model1)
model2_controller = ModelController(model2)
model3_controller = ModelController(model3)
metrics = evaluator.evaluate_traffic_controllers(
    [model1_controller, model2_controller, model3_controller, controller1, controller2]
)

for env_n in metrics[0].keys():
    print(env_n)
    for m in metrics:
        print(m[env_n])


#
# for i in range(100):
#     model = Frap()
#     model.load_state_dict(model_w['agent_state_dict']['model'])
#     model = model.eval()
#
#     model_controller = ModelController(model)
#     metrics = evaluator.evaluate_traffic_controllers([model_controller, controller1])
#
#     for m in metrics:
#         print(m)
#
#
#     if metrics[0]['random']['throughput'] > metrics[1]['random']['throughput']:
#         with open(Path(PROJECT_ROOT, 'saved', 'weird_random', 'better.pth'), 'wb') as f:
#             torch.save(model.state_dict(), f)
#         break
#
#     print("----")
# #
# env = SumoEnv.from_config_file(Path(JSONS_FOLDER, 'configs', 'example_test_all_equal.json'))
#
# with env.create_runner(render=True) as runner:
#     while True:
#         evaluate_controller(runner, model_controller, 50)
