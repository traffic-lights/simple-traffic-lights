#!/bin/bash

# chmod 777 collab_init.sh

sudo add-apt-repository ppa:sumo/stable -y
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc

pip install gym
pip install pygame
pip install dataclasses

# %env  SUMO_HOME=/usr/local/share/sumo
export SUMO_HOME=/usr/share/sumo # colab version
