import os, sys
from pathlib import Path
import argparse
from xml.dom import minidom
import json
from abc import ABC, abstractmethod

from settings import PROJECT_ROOT, JSONS_FOLDER

EVALUATOR_FOLDER = str(Path(JSONS_FOLDER, "evaluators"))
ENV_FOLDER = str(Path(JSONS_FOLDER, "envs"))


class Command(ABC):
    @abstractmethod
    def execute(self):
        pass

    @abstractmethod
    def help(self):
        pass


class ShowTests(Command):
    def __init__(self, gen):
        self.gen = gen

    def execute(self, args):
        print("current added tests:")
        for index, test_name in enumerate(self.gen.test_cases):
            print(f"{index}: {test_name}")

    def help(self):
        return ["st", "list all added tests"]


class AddTest(Command):
    def __init__(self, gen):
        self.gen = gen

    def execute(self, args):
        if len(args) != 3:
            print("arguments count missmatch, see help")
        else:
            self.gen.add_test_case(*args)

    def help(self):
        return [
            "at <name> <gen_type> <env_path>",
            "add test case",
            "- <name> - test name",
            "- <gen_type> - type of vehicle generator",
            "- <env_path> - path to sumocfg file",
        ]


class RemoveTest(Command):
    def __init__(self, gen):
        self.gen = gen

    def execute(self, args):
        if len(args) != 1:
            print("argument count missmatch, see help")
        else:
            try:
                del self.gen.test_cases[args[0]]
            except KeyError:
                print(f"{args[0]} does not exist")

    def help(self):
        return ["rt <name>", "remove test case", "- <name> - test case name to remove"]


class SaveAndQuit(Command):
    def __init__(self, gen):
        self.gen = gen

    def execute(self, args):
        if len(args) != 1:
            print("argument count missmatch, see help")
        else:
            self.gen.save(*args)
            self.gen.quit()

    def help(self):
        return [
            "sq <file_name>",
            "save and quit",
            "- <file_name> - test configuration file name",
        ]


class Quit(Command):
    def __init__(self, gen):
        self.gen = gen

    def execute(self, args):
        self.gen.quit()

    def help(self):
        return ["q", "quit without saving"]


class Help(Command):
    def __init__(self, commands):
        self.commands = commands

    def execute(self, args):
        print("available commands:")
        for _, command in self.commands.items():
            ret = command.help()
            print('{:<32s} {}'.format(ret[0], ret[1]))
            for s in ret[2:]:
                print('{:<35s} {}'.format("", s))

    def help(self):
        return ["h", "displays help"]


class Unknown(Command):
    def __init__(self):
        pass

    def execute(self, args):
        print("command not found, see help")

    def help(self):
        return ""


class TestCase:
    def __init__(self, name, gen_type, lanes, sumocfg):
        self.name = name
        self.gen_type = gen_type
        self.lanes = lanes
        self.sumocfg = sumocfg

        self.parameters = []

    def define_parameters(self):
        print("available lanes:")
        for index, lane in enumerate(self.lanes):
            print(f"{index}: {lane}")

        lanes = str(input("provide lane indexes for this test case: "))
        lanes = [self.lanes[i] for i in [int(id) for id in lanes.split(",")]]

        for lane in lanes:
            print(f"provide parameters for lane: {lane}")

            parameters = {"lane": lane, "active": True}

            if self.gen_type == "sin":
                parameters["amplitude"] = int(input("amplitude: "))
                parameters["multiplier"] = int(input("multiplier: "))
                parameters["start"] = int(input("start: "))
                parameters["min"] = int(input("min: "))
            elif self.gen_type == "rand":
                parameters["min_period"] = int(input("min_period: "))
                parameters["max_period"] = int(input("max_period: "))
            elif self.gen_type == "const":
                parameters["period"] = int(input("period: "))
            else:
                print(f"unknown gen_type: {self.gen_type}")

            self.parameters.append(parameters)


class Generator:
    def __init__(self):
        self.running = True
        self.test_cases = {}

    def add_test_case(self, name, gen_type, sumocfg_path):
        absolute_path = Path(PROJECT_ROOT, "environment", sumocfg_path)
        parent_folder = absolute_path.parent

        try:
            sumocfg = minidom.parse(str(absolute_path))
            
            sumorou = (
                sumocfg.getElementsByTagName("route-files")[0].attributes["value"].value
            )
            sumorou = str(Path(parent_folder, sumorou))

            sumonet = sumocfg.getElementsByTagName("net-file")[0].attributes["value"].value
            sumonet = str(Path(parent_folder, sumonet))

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

            self.test_cases[name] = TestCase(
                name, gen_type, sorted(start_lanes), sumocfg_path
            )
            self.test_cases[name].define_parameters()

        except FileNotFoundError:
            print("bad file path!")

    def save(self, file_name):
        test_cases = []
        for test_case_name in self.test_cases:
            test_case = self.test_cases[test_case_name]
            env_path = str(Path(ENV_FOLDER, f"{file_name}_{test_case_name}.json"))
            absolute_path = str(Path(PROJECT_ROOT, env_path))
            test_case_dict = {"name": test_case_name, "generator": env_path}
            test_cases.append(test_case_dict)

            test_case_dump = {
                "config_file": test_case.sumocfg,
                "vehicle_generator": {
                    "type": test_case.gen_type,
                    "lanes": test_case.parameters,
                },
            }

            with open(absolute_path, "w") as fp:
                json.dump(test_case_dump, fp, indent=4)

        config_dump = {"test_cases": test_cases}

        config_path = str(Path(PROJECT_ROOT, EVALUATOR_FOLDER, f"{file_name}.json"))
        with open(config_path, "w") as fp:
            json.dump(config_dump, fp, indent=4)

    def quit(self):
        self.running = False

    def run(self):
        commands = {
            "st": ShowTests(self),
            "at": AddTest(self),
            "rt": RemoveTest(self),
            "sq": SaveAndQuit(self),
            "q": Quit(self),
        }

        commands["h"] = Help(commands)

        unknown = Unknown()

        print("Welcome to test configuration generator.")
        commands["h"].execute([])
        while self.running:
            data = str(input("> "))
            data = data.split(" ")
            args = [] if len(data) <= 1 else data[1:]

            command = commands.get(data[0], unknown)
            command.execute(args)


if __name__ == "__main__":
    gen = Generator()
    gen.run()
