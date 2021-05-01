import os
import re

from setuptools import find_packages, setup

# define the main folder (in our case, this is "hub/")
project_name = "hub"


def get_property(prop, project):
    """
    this function allows you to get a property from the `project_name/__init__.py` file.

    if you call `get_property("__version__", "hub")`, it will return the __version__ variable you
    set inside `hub/__init__.py`.

    this means we can define the package version once & hub.__version__ will return it.
    """

    fname = project + "/__init__.py"
    result = re.search(
        # find variable with name `prop` in the `project + __init__.py` file.
        r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop),
        open(fname).read(),
    )
    return result.group(1)


this_directory = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(this_directory, "requirements.txt")) as f:
    requirements = f.readlines()

setup(
    name=project_name,
    version=get_property("__version__", project_name),
    description="",
    author="activeloop.ai",
    author_email="shashank@activeloop.ai",
    packages=find_packages(),
    install_requires=requirements,
)
