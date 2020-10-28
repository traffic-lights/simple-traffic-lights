import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
JSONS_FOLDER = Path(PROJECT_ROOT, 'jsons')
ENVIRONMENTS_FOLDER = Path(PROJECT_ROOT, 'environments')

SUMO_ROOT_PATH = os.path.join("/usr", "share", "sumo")
if "SUMO_HOME" not in os.environ:
    print("sumo home not in path")
    SUMO_TOOLS_PATH = os.path.join(SUMO_ROOT_PATH, "tools")
    os.environ['SUMO_HOME'] = SUMO_ROOT_PATH
else:
    SUMO_TOOLS_PATH = os.path.join(os.environ["SUMO_HOME"], "tools")

inited = False


def init_sumo_tools():
    global inited
    if not inited:
        inited = True
        sys.path.append(SUMO_TOOLS_PATH)
