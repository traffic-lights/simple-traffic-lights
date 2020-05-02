import os

from environment.env import SumoEnv

with SumoEnv() as env:
    env.reset()

    print(env.observation_space.shape)
    for i in range(5):
        a = env.action_space.sample()

        s, r, d = env.step(a)
        print(a, r)
