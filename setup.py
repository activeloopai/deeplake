import os
from os import path
from setuptools import find_packages, setup

project = "hub"
version = "0.1.5"

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md')) as f:
    long_description = f.read()

setup(
    name=project,
    version=version,
    description="Snark Hub",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Snark AI Inc.",
    author_email="support@snark.ai",
    url="https://github.com/snarkai/hub",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    keywords="snark-hub",
    python_requires='>=3',
    install_requires=[
        "click>=6.7,<7",
        "pathos==0.2.2.1",
        "boto3==1.9.2",
        "botocore==1.12.204",
        "numpy", 
        "tenacity"
    ],
    setup_requires=[],
    dependency_links=[],
    entry_points={
        "console_scripts": [
            "hub = hub:cli",
        ],
    },
    tests_require=[
        "pytest",
        "mock>=1.0.1",
    ],
)