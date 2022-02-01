import textwrap

from hub.client.config import HUB_REST_ENDPOINT
import click
from humbug.report import Report

from hub.client.client import HubBackendClient
from hub.client.utils import remove_username_from_config, write_token, remove_token
from hub.util.bugout_reporter import (
    save_reporting_config,
    get_reporting_config,
    hub_reporter,
)
from hub.util.exceptions import AuthenticationException


@click.command()
@click.option("--username", "-u", default=None, help="Your Activeloop Username")
@click.option("--password", "-p", default=None, help="Your Activeloop Password")
def login(username: str, password: str):
    """Log in to Activeloop"""
    chances: int = 3
    while chances:
        if not username:
            click.echo("Login to Activeloop using your credentials.")
            click.echo(
                "If you don't have an account, register by using the 'activeloop register' command or by going to "
                f"{HUB_REST_ENDPOINT}/register."
            )
            username = click.prompt("Username")
            password = click.prompt("Password", hide_input=True)
        if not password:
            password = click.prompt(
                f"Please enter password for user {username}", hide_input=True
            )
        username = username.strip()
        password = password.strip()
        try:
            client = HubBackendClient()
            token = client.request_auth_token(username, password)
            write_token(token)
            click.echo("Successfully logged in to Activeloop.")
            reporting_config = get_reporting_config()
            if reporting_config.get("username") != username:
                save_reporting_config(True, username=username)
            break
        except AuthenticationException:
            chances -= 1
            if chances:
                print("Login failed. Check username and password.")
                username = ""
                password = ""
            else:
                print(
                    "3 unsuccessful attempts. Kindly retry logging in after sometime."
                )
        except Exception as e:
            print(f"Encountered an error {e} Please try again later.")
            break


@click.command()
def logout():
    """Log out of Activeloop"""
    remove_token()
    remove_username_from_config()
    click.echo("Logged out of Activeloop.")


# TODO: Add how to enable/disable reporting to docs
@click.command()
@click.option("--on/--off", help="Turn crash report on/off")
def reporting(on):
    """Enable or disable sending crash report to Activeloop AI"""
    report = Report(
        title="Consent change",
        tags=hub_reporter.system_tags(),
        content=f"Consent? `{on}`",
    )
    hub_reporter.publish(report)
    save_reporting_config(on)


@click.command()
@click.option("--username", "-u", default=None, help="Your Activeloop Username")
@click.option("--email", "-e", default=None, help="Your Email")
@click.option("--password", "-p", default=None, help="Your Activeloop Password")
def register(username: str, email: str, password: str):
    """Create a new Activeloop user account"""
    click.echo("Thank you for registering for an Activeloop account!")
    if not username:
        click.echo(
            "Enter your details. Your password must be atleast 6 characters long."
        )
        username = click.prompt("Username")
        email = click.prompt("Email")
        password = click.prompt("Password", hide_input=True)
    if not password:
        password = click.prompt(
            f"Please enter the password you would like to associate with {username}",
            hide_input=True,
        )
    if not email:
        email = click.prompt(
            f"Please enter the Email Address you would like to associate with {username}"
        )
    username = username.strip()
    email = email.strip()
    password = password.strip()
    try:
        client = HubBackendClient()
        client.send_register_request(username, email, password)
        token = client.request_auth_token(username, password)
        write_token(token)
        click.echo(
            f"Successfully registered and logged in to Activeloop as {username}."
        )
        consent_message = textwrap.dedent(
            """
            Privacy policy:
            We collect basic system information and crash reports so that we can keep
            improving your experience using Hub to work with your data.
            You can find out more by reading our privacy policy:
                https://www.activeloop.ai/privacy/
            If you would like to opt out of reporting crashes and system information,
            run the following command:
                $ activeloop reporting --off
            """
        )
        click.echo(consent_message)
        save_reporting_config(True, username=username)
    except Exception as e:
        raise SystemExit(f"Unable to register new user: {e}")
