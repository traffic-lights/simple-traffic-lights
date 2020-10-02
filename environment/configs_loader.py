import json

from pathlib import Path
from settings import PROJECT_ROOT


CONFIGS_FILE = Path(PROJECT_ROOT, "environment", "configs", "environments.json")


def load_from_file(env_name):

    with open(CONFIGS_FILE) as file:
        data = json.load(file)

        assert env_name in data, f"{env_name} not defined in {CONFIGS_FILE}"

    return data[env_name]
