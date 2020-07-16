import click

from hub import config
from hub.log import configure_logger
from hub.cli.auth import login, logout, register


@click.group()
@click.option(
    "-h",
    "--host",
    default="{}".format(config.HUB_REST_ENDPOINT),
    help="Hub rest endpoint",
)
@click.option("-v", "--verbose", count=True, help="Devel debugging")
def cli(host, verbose):
    configure_logger(verbose)


def add_commands(cli):
    cli.add_command(login)
    cli.add_command(register)
    cli.add_command(logout)


add_commands(cli)
