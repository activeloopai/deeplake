from typing import *
import os, sys, time, random, uuid, traceback
from setuptools import find_packages, setup

project = "hub"
version = "0.4.1.5"
version = "0.5.0.0"

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md")) as f:
    long_description = f.read()

with open(os.path.join(this_directory, "requirements.txt"), "r") as f:
    requirements = f.readlines()

setup(
    name=project,
    version=version,
    description="Snark Hub",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Snark AI Inc.",
    author_email="support@snark.ai",
    url="https://github.com/snarkai/hub",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    keywords="snark-hub",
    python_requires=">=3",
    install_requires=requirements,
    setup_requires=[],
    dependency_links=[],
    entry_points={"console_scripts": ["hub = hub:cli",],},
    tests_require=["pytest", "mock>=1.0.1",],
)
