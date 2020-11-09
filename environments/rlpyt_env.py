import multiprocessing

from environments.sumo_env import SumoEnv
from rlpyt.envs.gym import GymEnvWrapper
from rlpyt.samplers.collections import TrajInfo


class AaaiTrajInfo(TrajInfo):
    """TrajInfo class for use with Atari Env, to store raw game score separate
    from clipped reward signal."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reward = 0
        self.pressure = 0
        self.travel_time = 0
        self.throughput = 0

    def step(self, observation, action, reward, done, agent_info, env_info):
        super().step(observation, action, reward, done, agent_info, env_info)
        self.reward = getattr(env_info, "reward", 0)
        self.pressure = getattr(env_info, "pressure", 0)
        self.travel_time = getattr(env_info, "travel_time", 0)
        self.throughput = getattr(env_info, "throughput", 0)


class PytConfig:

    def __init__(self, configs):
        if not isinstance(configs, list):
            configs = [configs]
        self.curr_env = multiprocessing.Value('i', 0)
        self.configs = configs

    def get_env(self):
        with self.curr_env.get_lock():
            conf = self.configs[self.curr_env.value]
            print(self.curr_env.value, conf, len(self.configs))
            self.curr_env.value = (self.curr_env.value + 1) % len(self.configs)

        return conf


class Rlpyt_env(GymEnvWrapper):
    def __init__(self,
                 pyt_conf,
                 max_steps=None
                 ):
        super().__init__(SumoEnv.from_config_file(pyt_conf.get_env(), max_steps=max_steps).create_runner(False))
