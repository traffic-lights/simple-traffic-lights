import multiprocessing

from environments.sumo_env import SumoEnv
from rlpyt.envs.gym import GymEnvWrapper
from rlpyt.samplers.collections import TrajInfo


class AaaiTrajInfo(TrajInfo):
    """TrajInfo class for use with Atari Env, to store raw game score separate
    from clipped reward signal."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, observation, action, reward, done, agent_info, env_info):
        super().step(observation, action, reward, done, agent_info, env_info)
        env_name = getattr(env_info, 'env_name', None)
        if env_name is not None:
            self._add_to_dict(env_info, '{}/'.format(env_name))
        else:
            self._add_to_dict(env_info)

    def _add_to_dict(self, env_info, prefix=''):
        self.__dict__['{}reward'.format(prefix)] = getattr(env_info, "reward", 0)
        self.__dict__['{}pressure'.format(prefix)] = getattr(env_info, "pressure", 0)
        self.__dict__['{}travel_time'.format(prefix)] = getattr(env_info, "travel_time", 0)
        self.__dict__['{}throughput'.format(prefix)] = getattr(env_info, "throughput", 0)


class PytConfig:

    def __init__(self, configs):
        if not (isinstance(configs, list) or isinstance(configs, dict)):
            configs = [configs]
        self.curr_env = multiprocessing.Value('i', 0)

        self.conf_keys = [None] * len(configs) if isinstance(configs, list) else list(configs.keys())
        self.conf_values = configs if isinstance(configs, list) else list(configs.values())

    def get_env(self):
        with self.curr_env.get_lock():
            conf_k = self.conf_keys[self.curr_env.value]
            conf_v = self.conf_values[self.curr_env.value]
            self.curr_env.value = (self.curr_env.value + 1) % len(self.conf_values)
        return conf_k, conf_v


class Rlpyt_env(GymEnvWrapper):
    def __init__(self,
                 pyt_conf,
                 max_steps=None
                 ):
        k, v = pyt_conf.get_env()

        super().__init__(SumoEnv.from_config_file(v, max_steps=max_steps, env_name=k).create_runner(False))
