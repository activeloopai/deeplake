#!/bin/bash

python3 -m venv .venv

# shellcheck source=/dev/null
source .venv/bin/activate

python3 setup_actions.py

curl -sSL https://install.python-poetry.org | python3 -

poetry install

pip install -r requirements.txt

python3 -m pytest test_activeloop*
