# Contributing Standards

## Linting
We use the [black](https://pypi.org/project/black/) python linter. You can have your code auto-formatted by running `pip install black`, then `black .` inside the directory you want to format.

## Docstrings
We use Google Docstrings. Please refer to [this example](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).

## Typing
We also use static typing for our function arguments/variables for better code readability. Our continuously integrated testing uses `pytest --mypy .` to run tests, which runs static type checking. 

The `--mypy` flag uses the [pytest-mypy](https://pypi.org/project/pytest-mypy/) pytest plugin.


