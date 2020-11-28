
from pathlib import Path
import pika
import json
from functools import partial
import argparse

from environments.sumo_env import SumoEnv
from settings import PROJECT_ROOT, JSONS_FOLDER


REPLY_TO = "amq.rabbitmq.reply-to"

parser = argparse.ArgumentParser("app to test component")
parser.add_argument("target")
parser.add_argument("--render", type=bool, default=True)
parser.add_argument("--max_steps", type=int, default=-1)


class Target:
    def __init__(self, file_name):
        with open(file_name) as json_file:
            data = json.load(json_file)

            self.address = data["address"]
            self.vhost = data["vhost"]
            self.requests_queue = data["requests_queue"]
            self.user = data["user"]
            self.password = data["password"]


def publish(state, routing):
    inp = {
        "last_action": int(state[0]),
        "in_states": [int(i) for i in state[1:]],
        "out_states": [0 for _ in range(12)]
    }

    channel.basic_publish(
        exchange='', routing_key=routing, body=json.dumps(inp), properties=pika.BasicProperties(reply_to=REPLY_TO))


def callback(env, publish_queue, ch, method, properties, body):
    data = json.loads(body)
    print(f"data: {data}")

    if data['code'] == 0:
        action = int(data['action'])
        state, _, done, _ = env.step(action)
    else:
        print(f"error: {data['code']} cause: {data['cause']}")
        done = False

    if done:
        env.close()
        ch.close()
        exit(0)

    publish(state, publish_queue)


if __name__ == "__main__":
    args = parser.parse_args()

    target_file = args.target
    max_steps = args.max_steps
    render = args.render

    print(f"render: {render}")

    target = Target(target_file)

    credentials = pika.PlainCredentials(target.user, target.password)
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host=target.address, port=5672, virtual_host=target.vhost, credentials=credentials))
    channel = connection.channel()

    env = SumoEnv.from_config_file(
        Path(JSONS_FOLDER, 'configs', 'aaai.json'), max_steps).create_runner(True)

    state = env.reset()

    callback = partial(callback, env, target.requests_queue)
    channel.basic_consume(REPLY_TO, callback, True)

    publish(state, target.requests_queue)

    channel.start_consuming()

    channel.close()
