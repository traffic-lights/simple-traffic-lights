import pygame

import numpy as np
from os import listdir
from os.path import isfile, join

from environment.dump import load_dumped


FPS = 8


def run(path):
    pygame.init()

    display = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Replayer")

    clock = pygame.time.Clock()

    running = False

    files = get_data(path)

    while not running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = True

        display.fill((255, 255, 255))

        draw_grid(display)

        try:
            phase, state = next(files)
        except StopIteration:
            break

        draw_cars(pos_list=state, surface=display)

        pygame.display.update()
        clock.tick(FPS)

    pygame.quit()


def draw_grid(surface):
    for y in np.arange(15.0, 615.0, 15.0):
        for x in np.arange(15.0, 615.0, 15.0):
            pygame.draw.line(surface, (0, 0, 0), (x, 0), (x, 600.0))
            pygame.draw.line(surface, (0, 0, 0), (0, y), (600.0, y))


def draw_cars(pos_list, surface):
    for pos in pos_list:
        x = pos[0]
        y = pos[1]
        pygame.draw.rect(surface, (255, 255, 0), (x * 15.0, y * 15.0, 15.0, 15.0))


def get_data(path):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    filenames = reversed([f for f in onlyfiles if ".resum" in f])

    for f in filenames:
        print(f"displaying {f}")
        yield load_dumped(f"{path}/{f}")
