from pathlib import Path
import json
from functools import partial
import asyncio
import aio_pika
import argparse
import math
import uuid
from datetime import datetime

from environments.sumo_env import SumoEnv
from settings import JSONS_FOLDER


parser = argparse.ArgumentParser("app to test component")
parser.add_argument("settings", help="client settings file name")
parser.add_argument("--max_steps", type=int, default=-1,
                    help="simulation max steps, -1==infinite")
parser.add_argument("--no_render", default=True,
                    action="store_false", help="render mode, default True")

RETRY_COUNT = 100
PUBLISH_TIMEOUT = 20

channel = None

current_timestamp = 0


class Settings:
    def __init__(self, file_name):
        with open(Path(JSONS_FOLDER, 'tester_settings', file_name)) as json_file:
            data = json.load(json_file)

            self.address = data["address"]
            self.vhost = data["vhost"]
            self.requests_queue = data["requests_queue"]
            self.responses_queue = data["responses_queue"]
            self.responses_exchange = data["responses_exchange"]
            self.user = data["user"]
            self.password = data["password"]
            self.env_file = data["env"]
            self.uuid = str(uuid.uuid1())


async def publish(state, settings, timestamp):
    inp = {
        "last_action": int(state[0]),
        "in_states": [int(i) for i in state[1:]],
        "out_states": [0 for _ in range(12)],
        "timestamp": str(timestamp)
    }

    return await channel.default_exchange.publish(
        aio_pika.Message(
            body=json.dumps(inp).encode(), reply_to=settings.uuid
        ),
        routing_key=settings.requests_queue,
        timeout=PUBLISH_TIMEOUT,
    )


async def process_message(settings, message: aio_pika.IncomingMessage):
    with message.process(requeue=True, ignore_processed=True):
        global current_timestamp
        data = json.loads(message.body.decode())
        print(f"data: {data}")

        received_timestamp = float(data["timestamp"])

        if received_timestamp >= current_timestamp:
            if data['code'] == 0:
                action = int(data['action'])
                state, _, done, _ = env.step(action)
            else:
                print(f"error: {data['code']} cause: {data['cause']}")
                done = False

            if done:
                exit(0)

            timestamp = datetime.timestamp(datetime.now())
            await publish(state, settings, timestamp)
            current_timestamp = max(timestamp, current_timestamp)
        else:
            print("received old message, skipping ...")
            print(
                f'timestamp: {data["timestamp"]} waiting for: {current_timestamp}')


async def main(loop, settings):
    global channel
    global current_timestamp
    url = f'amqp://{settings.user}:{settings.password}@{settings.address}/{settings.vhost}'
    connection = await aio_pika.connect_robust(
        url=url, loop=loop
    )

    channel = await connection.channel()

    queue = await channel.declare_queue(name=settings.uuid, auto_delete=False, durable=True, arguments={"x-queue-type": "quorum"})

    consumer = partial(process_message, settings)

    await queue.consume(consumer, no_ack=False)
    await publish(state, settings, current_timestamp)
    return connection


if __name__ == "__main__":
    global state
    global env

    args = parser.parse_args()

    settings_file = args.settings
    max_steps = args.max_steps
    render = args.no_render

    settings = Settings(settings_file)

    print(settings.uuid)

    env = SumoEnv.from_config_file(
        Path(JSONS_FOLDER, 'configs', settings.env_file), max_steps=max_steps).create_runner(render=render)

    state = env.reset()

    loop = asyncio.get_event_loop()
    connection = loop.run_until_complete(main(loop, settings))

    try:
        loop.run_forever()
    finally:
        loop.run_until_complete(connection.close())
