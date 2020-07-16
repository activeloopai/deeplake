import click

from hub import config
from hub.log import configure_logger
from hub.cli.command import add_commands


@click.group()
@click.option("-v", "--verbose", count=True, help="Devel debugging")
def cli(verbose):
    config.HUB_REST_ENDPOINT = "http://127.0.0.1:8085"
    configure_logger(verbose)


add_commands(cli)
