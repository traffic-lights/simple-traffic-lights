from pathlib import Path

from environment.env import SumoEnv
from settings import PROJECT_ROOT

from gym import error, spaces, utils
import traci
import traci.constants as tc

TRAFFICLIGHTS_PHASES = 4



class AaaiEnv(SumoEnv):
    def __init__(self,
                 config_file=Path(PROJECT_ROOT, "environment", "2lane.sumocfg"),
                 replay_folder=Path(PROJECT_ROOT, "replays"),
                 save_replay=False,
                 render=False):
        super().__init__(
            config_file=config_file,
            replay_folder=replay_folder,
            save_replay=save_replay,
            render=render
        )

        self.observation_space = spaces.Space(shape=(TRAFFICLIGHTS_PHASES + 1,))
        self.action_space = spaces.Discrete(1)
        self.tls_id = traci.trafficlight.getIDList()[0]

    def _snap_state(self):
        pass

    def _take_action(self, action):
        pass





    def _reset(self):
        pass
