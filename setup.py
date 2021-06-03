import os
import re

from setuptools import find_packages, setup

project_name = "hub"


def get_property(prop, project):
    """
    Get a property from the `project_name/__init__.py` file.

    Example:
        `get_property("__version__", "hub")`,
        returns the __version__ variable set inside `hub/__init__.py`.
    """

    fname = project + "/__init__.py"
    result = re.search(
        # find variable with name `prop` in the `project + __init__.py` file.
        r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop),
        open(fname).read(),
    )
    return result.group(1)


this_directory = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(this_directory, "requirements/common.txt")) as f:
    requirements = f.readlines()
setup(
    name=project_name,
    version=get_property("__version__", project_name),
    description="",
    author="activeloop.ai",
    author_email="shashank@activeloop.ai",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "hub = hub.cli.commands:cli",
        ]
    },
)
