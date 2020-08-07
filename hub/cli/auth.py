import click
from hub.log import logger
from hub import config
from hub.client.auth import AuthClient
from hub.client.token_manager import TokenManager
from hub.client.hub_control import HubControlClient


@click.command()
# @click.option('--token', is_flag=True, default=False,
#        help='Enter authentication tocken from {}'.format(config.GET_TOKEN_REST_SUFFIX))
@click.option("--username", "-u", default=None, help="Your Snark AI username")
@click.option("--password", "-p", default=None, help="Your Snark AI password")
def login(username, password):
    """ Logs in to Snark AI"""
    token = ""
    if token:
        logger.info("Token login.")
        logger.degug("Getting the token...")
        token = click.prompt(
            "Please paste the authentication token from {}".format(
                config.GET_TOKEN_REST_SUFFIX, type=str, hide_input=True
            )
        )
        token = token.strip()
        AuthClient.check_token(token)
    else:
        logger.info(
            "Please log in using Snark AI credentials. You can register at https://app.activeloop.ai "
        )
        if not username:
            logger.debug("Prompting for username.")
            username = click.prompt("Username", type=str)
        username = username.strip()
        if not password:
            logger.debug("Prompting for password.")
            password = click.prompt("Password", type=str, hide_input=True)
        password = password.strip()
        token = AuthClient().get_access_token(username, password)
    TokenManager.set_token(token)
    HubControlClient().get_credentials()
    logger.info("Login Successful.")


@click.command()
def logout():
    """ Logs out of Activeloop AI"""
    TokenManager.purge_token()


@click.command()
@click.option("--username", "-u", default=None, help="Your Activeloop AI username")
@click.option("--email", "-e", default=None, help="Your email")
@click.option("--password", "-p", default=None, help="Your Activeloop AI password")
def register(username, email, password):
    """ Register at of Activeloop AI"""
    if not username:
        logger.debug("Prompting for username.")
        username = click.prompt("Username", type=str)
    username = username.strip()

    if not email:
        logger.debug("Prompting for email.")
        email = click.prompt("Email", type=str)
    email = email.strip()

    if not password:
        logger.debug("Prompting for password.")
        password = click.prompt("Password", type=str, hide_input=True)
    password = password.strip()

    AuthClient().register(username, email, password)
    token = AuthClient().get_access_token(username, password)
    TokenManager.set_token(token)