from hub.client.config import HUB_REST_ENDPOINT
import click

from hub.client.client import HubBackendClient
from hub.client.utils import write_token, remove_token
from hub.util.exceptions import AuthenticationException


@click.command()
@click.option("--username", "-u", default=None, help="Your Activeloop username")
@click.option("--password", "-p", default=None, help="Your Activeloop password")
def login(username: str, password: str):
    """Log in to Activeloop"""
    click.echo("Login to Activeloop Hub using your credentials.")
    click.echo(
        "If you don't have an account, register by using 'activeloop register' command or by going to "
        f"{HUB_REST_ENDPOINT}/register."
    )
    username = username or click.prompt("Username")
    username = username.strip()
    password = password or click.prompt("Password", hide_input=True)
    password = password.strip()
    try:
        client = HubBackendClient()
        token = client.request_auth_token(username, password)
        write_token(token)
        click.echo("\nSuccessfully logged in to Activeloop Hub.")
    except AuthenticationException:
        raise SystemExit("\nLogin failed. Check username and password.")
    except Exception as e:
        raise SystemExit(f"\nUnable to login: {e}")


@click.command()
def logout():
    """Log out of Activeloop"""
    remove_token()
    click.echo("Logged out of Hub.")


@click.command()
@click.option("--username", "-u", default=None, help="Your Activeloop username")
@click.option("--email", "-e", default=None, help="Your email")
@click.option("--password", "-p", default=None, help="Your Activeloop password")
def register(username: str, email: str, password: str):
    """Create a new Activeloop user account"""
    click.echo("Enter your details. Your password must be atleast 6 characters long.")
    username = username or click.prompt("Username")
    username = username.strip()
    email = email or click.prompt("Email")
    email = email.strip()
    password = password or click.prompt("Password", hide_input=True)
    password = password.strip()
    try:
        client = HubBackendClient()
        client.send_register_request(username, email, password)
        token = client.request_auth_token(username, password)
        write_token(token)
        click.echo(
            f"\nSuccessfully registered and logged in to Activeloop Hub as {username}."
        )
    except Exception as e:
        raise SystemExit(f"\nUnable to register new user: {e}")
