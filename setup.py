"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import os

from setuptools import find_packages, setup

project = "hub_v1"


this_directory = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(this_directory, project, "version.py")) as f:
    exec(f.read())


with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

with open(os.path.join(this_directory, "requirements.txt")) as f:
    requirements = f.readlines()

setup(
    name=project,
    version=__version__,
    description="Activeloop Hub",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    author="Snark AI Inc.",
    author_email="support@activeloop.ai",
    license="MPL 2.0",
    url="https://github.com/activeloopai/Hub",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    keywords="snark-hub",
    project_urls={
        "Documentation": "https://docs.activeloop.ai/",
        "Source": "https://github.com/activeloopai/Hub",
    },
    classifiers=[
        "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3 :: Only",
    ],
    python_requires=">=3.6",
    install_requires=requirements,
    setup_requires=[],
    dependency_links=[],
    entry_points={
        "console_scripts": [
            "activeloop = hub.cli.command:cli",
            "activeloop-local = hub.cli.local:cli",
            "activeloop-dev = hub.cli.dev:cli",
            "hub = hub.cli.command:cli",
        ]
    },
    tests_require=["pytest", "mock>=1.0.1"],
)
