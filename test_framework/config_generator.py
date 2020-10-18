import os, sys
from pathlib import Path
import argparse
from xml.dom import minidom
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]

class TestCase:
    def __init__(self, name, gen_type, lanes):
        self.name = name
        self.gen_type = gen_type
        self.lanes = []
        for lane in lanes:
            self.lanes.append({"lane": lane})

    def define_parameters(self):
        for lane in self.lanes:
            print(f"provide parameters for lane: {lane['lane']}")
            lane["active"] = True
            if self.gen_type == "sin":
                lane["amplitude"] = int(input("amplitude: "))
                lane["multiplier"] = int(input("multiplier: "))
                lane["start"] = int(input("start: "))
                lane["min"] = int(input("min: "))
            elif self.gen_type == "rand":
                lane["min_period"] = int(input("min_period: "))
                lane["max_period"] = int(input("max_period: "))
            elif self.gen_type == "const":
                lane["period"] = int(input("period: "))
            else:
                print(f"unknown gen_type: {self.gen_type}")


class Generator:
    def __init__(self, lanes, config_file):
        self.lanes = sorted(lanes)
        self.running = True
        self.test_cases = {}
        self.config_file = config_file

    def display_available_lanes(self):
        for index, lane in enumerate(self.lanes):
            print(f"{index} : {lane}")

    def add_test_case(self, name, gen_type, lanes):
        lanes = [self.lanes[i] for i in [int(id) for id in lanes.split(",")]]
        self.test_cases[name] = TestCase(name, gen_type, lanes)
        self.test_cases[name].define_parameters()

    def save(self, config_path, generator_path, file_name):
        test_cases = []
        for test_case_name in self.test_cases:
            test_case = self.test_cases[test_case_name]
            test_case_path = f"{generator_path}/{file_name}_{test_case_name}.json"
            test_case_dict = {"name": test_case_name, "generator": test_case_path}
            test_cases.append(test_case_dict)

            test_case_dump = {
                "config_file": self.config_file,
                "replay_folder": "replays",
                "vehicle_generator": {
                    "type": test_case.gen_type,
                    "lanes": test_case.lanes,
                },
            }

            with open(test_case_path, "w") as fp:
                json.dump(test_case_dump, fp, indent=4)

        config_dump = {"test_cases": test_cases}

        with open(f"{config_path}/{file_name}.json", "w") as fp:
            json.dump(config_dump, fp, indent=4)

    def quit(self):
        self.running = False


if __name__ == "__main__":
    print("Enter env name:")
    env_name = str(input())
    sumocfg = f'{env_name}/{env_name}.sumocfg'
    sumorou = f'{env_name}/{env_name}.rou.xml'
    sumorou = str(Path(PROJECT_ROOT, "environment", sumorou))
    sumonet = f'{env_name}/{env_name}.net.xml'
    sumonet = str(Path(PROJECT_ROOT, "environment", sumonet))

    sumorou = minidom.parse(sumorou)

    start_edges = set()
    for route in sumorou.getElementsByTagName("route"):
        route = route.attributes["edges"].value
        start_edges.add(route.split(" ")[0])

    sumonet = minidom.parse(sumonet)

    start_lanes = set()
    for edge in sumonet.getElementsByTagName("edge"):
        edge_id = edge.attributes["id"].value
        if edge_id in start_edges:
            for lane in edge.getElementsByTagName("lane"):
                lane_id = lane.attributes["id"].value
                start_lanes.add(lane_id)

    gen = Generator(start_lanes, sumocfg)

    commands_message = """Available commands: 
                    - al - list all available lanes
                    - at - add test case
                    - sq - save and quit
                    - h - displays help"""

    print("Welcome to test configuration generator.")
    print(commands_message)
    while gen.running:
        data = str(input("> "))
        data = data.split(" ")
        command = data[0]
        args = [] if len(data) <= 1 else data[1:]

        if command == "al":
            gen.display_available_lanes()
        elif command == "at":
            if len(args) != 3:
                print("no enough arguments: at <name> <gen_type> <lanes>")
            else:
                gen.add_test_case(*args)
        elif command == "sq":
            if len(args) != 3:
                print(
                    "no enough arguments: sq <config_path> <generator_path> <file_name>"
                )
            else:
                gen.save(*args)
                gen.quit()
        elif command == "h":
            print(commands_message)
        else:
            print(f"unknown command '{command}'")
            print(commands_message)
