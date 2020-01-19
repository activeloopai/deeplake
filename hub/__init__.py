
from .creds import Base as Creds
from .array import Array
from .bucket import Bucket
from .dataset import Dataset

s3 = Creds._s3
gs = Creds._gs
fs = Creds._fs

# @click.group()
# @click.option('-v', '--verbose', count=True, help='Devel debugging')
# def cli(verbose):
#     configure_logger(verbose)

# def add_commands(cli):
#     cli.add_command(configure)

# add_commands(cli)
