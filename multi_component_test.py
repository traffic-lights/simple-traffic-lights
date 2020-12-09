from pathlib import Path
import json
from functools import partial
import asyncio
import aio_pika
import argparse
import math
import uuid
import nest_asyncio
from datetime import datetime

from environments.sumo_env import SumoEnv
from settings import JSONS_FOLDER


parser = argparse.ArgumentParser("app to test component")
parser.add_argument("settings", help="client settings file name")
parser.add_argument("--max_steps", type=int, default=-1,
                    help="simulation max steps, -1==infinite")
parser.add_argument("--no_render", default=True,
                    action="store_false", help="render mode, default True")

parser.add_argument("--verbose", default=False,
                    action="store_true", help="displays debug information")

RETRY_COUNT = 100
PUBLISH_TIMEOUT = 20

channel = None


class JunctionQueue:
    def __init__(self):
        self.queue = []
        self.timestamp = 0

    def push(self, action, timestamp):
        if timestamp >= self.timestamp:
            self.queue.append(action)
            self.timestamp = timestamp
        else:
            print("received old message, skipping ...")
            print(
                f'timestamp: {timestamp} waiting for: {self.timestamp}')

    def get(self):
        res = self.queue[-1]
        self.queue = []
        return res

    def not_empty(self):
        return len(self.queue) > 0


class Settings:
    def __init__(self, file_name):
        with open(Path(JSONS_FOLDER, 'tester_settings', file_name)) as json_file:
            data = json.load(json_file)

            self.address = data["address"]
            self.vhost = data["vhost"]
            self.requests_queue = data["requests_queue"]
            self.responses_queue = data["responses_queue"]
            self.user = data["user"]
            self.password = data["password"]
            self.env_file = data["env"]

            self.verbose = False


async def publish(state, settings, timestamp, id):
    local_state = state[0]
    inp = {
        "last_action": int(local_state[0]),
        "in_states": [int(i) for i in local_state[1:]],
        "out_states": [0 for _ in range(12)],
        "timestamp": str(timestamp),
        "junction_id": str(id)
    }

    return await channel.default_exchange.publish(
        aio_pika.Message(
            body=json.dumps(inp).encode(), reply_to=settings.responses_queue
        ),
        routing_key=settings.requests_queue,
        timeout=PUBLISH_TIMEOUT,
    )


async def send(junction_queues, state, settings):
    counter = 0
    for i, junction in junction_queues.items():
        timestamp = datetime.timestamp(datetime.now())
        await publish(state, settings, timestamp, str(i))
        junction.timestamp = max(timestamp, junction_queues[i].timestamp)
        counter += 1

    return counter == len(junction_queues)


async def process_message(settings, message: aio_pika.IncomingMessage):
    with message.process(requeue=True, ignore_processed=True):
        global junction_queues

        data = json.loads(message.body.decode())
        if settings.verbose:
            print(f"data: {data}")

        received_timestamp = float(data["timestamp"])

        if data['code'] == 0:
            junction = int(data['junction_id'])
            junction_queues[junction].push(
                int(data['action']), received_timestamp)
        else:
            print(f"error: {data['code']} cause: {data['cause']}")

        ready = True
        for junction in junction_queues.values():
            ready = ready and junction.not_empty()

        if ready:
            actions = [j.get() for j in junction_queues.values()]
            state, _, done, _ = env.step(actions)

            if done:
                exit(0)

            await send(junction_queues, state, settings)


async def reconnect_callback(settings):
    global junction_queues
    global state

    sent = False
    while not sent:
        sent = await send(junction_queues, state, settings)


async def main(loop, settings):
    global channel
    url = f'amqp://{settings.user}:{settings.password}@{settings.address}/{settings.vhost}'
    connection = await aio_pika.connect_robust(
        url=url, loop=loop
    )

    def callback_wrapped(a, b): return asyncio.run(
        reconnect_callback(settings))

    connection.add_reconnect_callback(callback_wrapped)

    channel = await connection.channel()

    queue = await channel.get_queue(settings.responses_queue)

    consumer = partial(process_message, settings)

    await queue.consume(consumer, no_ack=False)

    await send(junction_queues, state, settings)

    return connection


if __name__ == "__main__":
    global state
    global env
    global junction_queues

    nest_asyncio.apply()

    args = parser.parse_args()

    settings_file = args.settings
    max_steps = args.max_steps
    render = args.no_render
    verbose = args.verbose

    settings = Settings(settings_file)
    settings.verbose = verbose

    env = SumoEnv.from_config_file(
        Path(JSONS_FOLDER, 'configs', settings.env_file), max_steps=max_steps).create_runner(render=render)

    state = env.reset()

    junction_queues = {i: JunctionQueue() for i in range(len(state))}

    loop = asyncio.get_event_loop()
    connection = loop.run_until_complete(main(loop, settings))

    try:
        loop.run_forever()
    finally:
        loop.run_until_complete(connection.close())
