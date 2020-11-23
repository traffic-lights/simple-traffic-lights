
from pathlib import Path
import pika
import json
from functools import partial

from environments.sumo_env import SumoEnv
from settings import PROJECT_ROOT, JSONS_FOLDER


REPLY_TO = "amq.rabbitmq.reply-to"


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
        state, _, _, _ = env.step(action)
    else:
        print(f"error: {data['code']} cause: {data['cause']}")

    publish(state, publish_queue)


if __name__ == "__main__":
    address = str(input("address: "))
    model = str(input("model name: "))

    credentials = pika.PlainCredentials(f'{model}_client', 'tajnehaslo')
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host=address, port=5672, virtual_host="traffily", credentials=credentials))
    channel = connection.channel()

    env = SumoEnv.from_config_file(
        Path(JSONS_FOLDER, 'configs', 'aaai.json')).create_runner(True)

    state, _, _, _ = env.step(0)

    callback = partial(callback, env, f"{model}_requests")
    channel.basic_consume(REPLY_TO, callback, True)

    publish(state, f"{model}_requests")

    channel.start_consuming()

    channel.close()
