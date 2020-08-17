from time import sleep

from environment.aaai_env import AaaiEnv

with AaaiEnv(render=True) as env:
    sleep(10)

    for i in range(8):
        print("changing to {}".format(i))
        print('throughput: {}, travel_time: {}'.format(env.get_throughput(), env.get_travel_time()))
        env.step(i)
