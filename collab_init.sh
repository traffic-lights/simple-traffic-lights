#!/bin/bash

sudo add-apt-repository ppa:sumo/stable -y
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc

pip install traci
pip install gym

export SUMO_HOME=/usr/local/share/sumo