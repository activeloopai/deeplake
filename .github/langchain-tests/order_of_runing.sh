#!/bin/bash

python3 setup_actions.py

curl -sSL https://install.python-poetry.org | python3 -
poetry shell
source .venv/bin/activate
poetry install

pip install -r requirements.txt
python3 -m pytest test_activeloop*
