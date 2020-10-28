from environments.aaai_env import AaaiEnv
from environments.simple_env import SimpleEnv

ENVIRONMENTS_TYPE_MAPPER = {
    'aaai': AaaiEnv,
    'simple': SimpleEnv,
}
