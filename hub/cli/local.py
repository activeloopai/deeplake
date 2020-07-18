import click

from hub import config
from hub.log import configure_logger
from hub.cli.command import add_commands


@click.group()
@click.option("-v", "--verbose", count=True, help="Devel debugging")
def cli(verbose):
    config.HUB_REST_ENDPOINT = config.HUB_LOCAL_REST_ENDPOINT
    configure_logger(verbose)


add_commands(cli)
