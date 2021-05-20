from hub.client.utils import write_token, remove_token
from hub.client.client import HubBackendClient
import click


@click.command()
@click.option("--username", "-u", default=None, help="Your Activeloop username")
@click.option("--password", "-p", default=None, help="Your Activeloop password")
def login(username, password):
    """Log in to Activeloop"""

    click.echo("Log in using Activeloop credentials.")
    click.echo(
        "If you don't have an account register by using 'hub register' command or by going to https://app.activeloop.ai/register"
    )
    username = username or click.prompt("Username", type=str)
    username = username.strip()
    password = password or click.prompt("Password", type=str, hide_input=True)
    password = password.strip()
    token = HubBackendClient().request_access_token(username, password)
    write_token(token)
    click.echo("Successfully logged in to Hub.")


@click.command()
def logout():
    """Log out of Activeloop"""

    remove_token()
    click.echo("Logged out of Hub.")


@click.command()
@click.option("--username", "-u", default=None, help="Your Activeloop username")
@click.option("--email", "-e", default=None, help="Your email")
@click.option("--password", "-p", default=None, help="Your Activeloop password")
def register(username, email, password):
    """Create a new Activeloop user account"""

    click.echo("Enter your details. Your password must be atleast 6 characters long.")
    username = username or click.prompt("Username", type=str)
    username = username.strip()
    email = email or click.prompt("Email", type=str)
    email = email.strip()
    password = password or click.prompt("Password", type=str, hide_input=True)
    password = password.strip()
    HubBackendClient().send_register_request(username, email, password)
    token = HubBackendClient().request_access_token(username, password)
    write_token(token)
