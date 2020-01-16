import hub
from hub.marray.interface import array, load, dataset, S3, GS, FS
from hub.marray.dataset import Dataset
import click
import sys
from hub.log import configure_logger
from .cli.auth import configure

from .api.api import amazon_s3 as s3
from .api.api import filesystem as fs
from .api.api import bucket

@click.group()
@click.option('-v', '--verbose', count=True, help='Devel debugging')
def cli(verbose):
    configure_logger(verbose)

def add_commands(cli):
    cli.add_command(configure)

add_commands(cli)
