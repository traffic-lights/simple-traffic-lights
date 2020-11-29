from pathlib import Path
import pika
import json
from functools import partial
import argparse
from time import sleep
import math

from environments.sumo_env import SumoEnv
from settings import PROJECT_ROOT, JSONS_FOLDER

REPLY_TO = "amq.rabbitmq.reply-to"

parser = argparse.ArgumentParser("app to test component")
parser.add_argument("settings", help="client settings file name")
parser.add_argument("--max_steps", type=int, default=-1, help="simulation max steps, -1==infinite")
parser.add_argument("--no_render", default=True, action="store_false", help="render mode, default True")


class Settings:
    def __init__(self, file_name):
        with open(Path(PROJECT_ROOT, 'tester_settings', file_name)) as json_file:
            data = json.load(json_file)

            self.address = data["address"]
            self.vhost = data["vhost"]
            self.requests_queue = data["requests_queue"]
            self.user = data["user"]
            self.password = data["password"]
            self.env_file = data["env"]


def publish(channel, state, routing):
    inp = {
        "last_action": int(state[0]),
        "in_states": [int(i) for i in state[1:]],
        "out_states": [0 for _ in range(12)]
    }

    channel.basic_publish(
        exchange='', routing_key=routing, body=json.dumps(inp),
        properties=pika.BasicProperties(reply_to=REPLY_TO)
    )


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
        ch.stop_consuming()
    else:
        publish(ch, state, publish_queue)


if __name__ == "__main__":
    args = parser.parse_args()

    settings_file = args.settings
    max_steps = args.max_steps
    render = args.no_render

    settings = Settings(settings_file)

    credentials = pika.PlainCredentials(settings.user, settings.password)

    env = SumoEnv.from_config_file(
        Path(JSONS_FOLDER, 'configs', settings.env_file), max_steps).create_runner(render=render)

    state = env.reset()
    callback = partial(callback, env, settings.requests_queue)

    retries = 0
    while True:
        if retries > 5:
            retries = 5

        backoff = 0.05 * math.exp(retries)
        try:
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(host=settings.address, port=5672, virtual_host=settings.vhost, credentials=credentials))
            channel = connection.channel()

            channel.basic_consume(REPLY_TO, callback, True)

            publish(channel, state, settings.requests_queue)
            retries = 0
        except Exception as e:
            print("Unable to connect. Retrying ...")
            sleep(backoff)
            retries += 1
            continue

        try:
            channel.start_consuming()
        except Exception as e:
            print(f"DEBUG: {e}")
            print("Connection closed. Reconnecting ...")
            backoff = 0.05 * math.exp(retries)
            sleep(backoff)
            retries += 1
            continue
        except KeyboardInterrupt:
            channel.stop_consuming()

        channel.close()
        break
