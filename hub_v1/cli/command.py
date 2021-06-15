"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import click
from hub_v1 import config
from hub_v1.log import configure_logger
from hub_v1.cli.auth import login, logout, register, reporting
from hub_v1.version import __version__


@click.group()
@click.option(
    "-h",
    "--host",
    default=f"{config.HUB_REST_ENDPOINT}",
    help="Hub rest endpoint",
)
@click.option("-v", "--verbose", count=True, help="Devel debugging")
def cli(host, verbose):
    configure_logger(verbose)


@click.group()
@click.version_option(__version__)
@click.pass_context
def cli(ctx):
    pass


def add_commands(cli):
    cli.add_command(login)
    cli.add_command(register)
    cli.add_command(logout)
    cli.add_command(cli)
    cli.add_command(reporting)


add_commands(cli)
