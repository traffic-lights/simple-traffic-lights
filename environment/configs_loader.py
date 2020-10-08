import json

from pathlib import Path
from settings import PROJECT_ROOT


def load_from_file(config_file_path):
    config_flie_path = Path(PROJECT_ROOT, "environment", "configs", config_file_path)

    with open(config_flie_path) as file:
        data = json.load(file)

        return data
