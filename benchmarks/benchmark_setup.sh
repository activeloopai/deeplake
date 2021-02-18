#!/bin/bash
sudo yum -y install python3
python3 -m venv hub-env
source ./hub-env/bin/activate
pip install pip --upgrade
sudo yum -y install git
git clone https://github.com/activeloopai/Hub
pip install -e ./Hub/
pip install -r ./Hub/requirements-dev.txt
pip install -r ./Hub/requirements-optional.txt
pip install -r ./Hub/requirements-benchmarks.txt
