"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import click

from hub import config
from hub.log import configure_logger
from hub.cli.command import add_commands


@click.group()
@click.option("-v", "--verbose", count=True, help="Devel debugging")
def cli(verbose):
    config.HUB_REST_ENDPOINT = config.HUB_DEV_REST_ENDPOINT
    configure_logger(verbose)


add_commands(cli)
