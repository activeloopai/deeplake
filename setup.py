from hub.util.get_property import get_property
import os

from setuptools import find_packages, setup

project_name = "hub"


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
            "activeloop = hub.cli.commands:cli",
        ]
    },
)
