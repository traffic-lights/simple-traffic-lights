import torch

from pathlib import Path

from settings import PROJECT_ROOT, JSONS_FOLDER
from trainings.training_parameters import TrainingState
from models.frap_production import Frap 

        
if __name__ == "__main__":
    path = str(input("provide path to attributes: "))
    name = str(input("provide file name to save: "))

    model = Frap()

    model_w = torch.load(path, map_location='cpu')
    model.load_state_dict(model_w['agent_state_dict']['model'])
    model = model.eval()

    m = torch.jit.script(model)
    model_path = Path(PROJECT_ROOT, "cpp_models", f'{name}.pt')
    print(f'model path: {model_path}')
    torch.jit.save(m, str(model_path))