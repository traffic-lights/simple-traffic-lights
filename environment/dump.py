import gzip
import json
import numpy as np


def convert_to_list(matrix):
    res = []
    for y in range(40):
        for x in range(40):
            if matrix[0][x][y] == 1:
                res.append((x, y))

    return res


def dump_to_file(data, path="dump.resum"):
    phase, state = data

    resum_file = {"phase": phase, "state": convert_to_list(state)}

    dump_file = json.dumps(resum_file)

    with gzip.open(path, "wb") as f:
        f.write(dump_file.encode())


def load_dumped(path):
    phase, state = None, None
    with gzip.open(path, "r") as f:
        content = f.read()

        content = json.loads(content.decode())
        phase = content["phase"]
        state = content["state"]

    return phase, state
