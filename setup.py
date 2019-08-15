import os
from setuptools import find_packages, setup

project = "hub"
version = "0.1.3"

setup(
    name=project,
    version=version,
    description="Snark Hub",
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