from time import sleep

from environment.aaai_env import AaaiEnv

with AaaiEnv(render=True) as env:
    sleep(10)

    for i in range(8):
        print("zmieniam na {}".format(i))
        env.step(i)

    #
    # print("zmieniam na 0")
    # print(env.step(0))
    # print("zmieniam na 0")
    # print(env.step(0))
    # print("zmieniam na 1")
    # print(env.step(1))
    # print("zmieniam na 1")
    # print(env.step(1))
    # print('throughput: {}, travel_time: {}'.format(env.get_throughput(), env.get_travel_time()))
    # print("zmieniam na 2")
    # print(env.step(2))
    # print("zmieniam na 2")
    # print(env.step(2))
    # print('throughput: {}, travel_time: {}'.format(env.get_throughput(), env.get_travel_time()))
    # print("zmieniam na 3")
    # print(env.step(3))
    # print("zmieniam na 3")
    # print(env.step(3))
    # print("zmieniam na 0")
    # print(env.step(0))
    # print("zmieniam na 0")
    # print(env.step(0))
    # print("zmieniam na 2")
    # print(env.step(2))
    #
    # print('throughput: {}, travel_time: {}'.format(env.get_throughput(), env.get_travel_time()))
    #
    # for i in range(5):
    #     print(env.step(0))