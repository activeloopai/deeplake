import click

from deeplake import __version__
from deeplake.cli.auth import login, logout, register, reporting


@click.group()
@click.version_option(__version__, message="%(prog)s %(version)s")
@click.pass_context
def cli(ctx):
    pass


def add_auth_commands(cli):
    cli.add_command(login)
    cli.add_command(logout)
    cli.add_command(register)
    cli.add_command(reporting)


add_auth_commands(cli)
