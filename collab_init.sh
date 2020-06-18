#!/bin/bash

# chmod 777 collab_init.sh

sudo add-apt-repository ppa:sumo/stable -y
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc


# pip install traci
pip install gym
pip install pygame

# %env  SUMO_HOME=/usr/local/share/sumo
export SUMO_HOME=/usr/local/share/sumo
