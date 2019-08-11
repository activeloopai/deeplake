import os
from setuptools import find_packages, setup

project = "hub"
version = "0.0.1"

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
        "cloud-volume==0.56.2"
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