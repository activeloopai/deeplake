"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
import textwrap

import click
from humbug.report import Report

from hub_v1.log import logger
from hub_v1 import config
from hub_v1.client.auth import AuthClient
from hub_v1.client.token_manager import TokenManager
from hub_v1.client.hub_control import HubControlClient
from hub_v1.report import (
    configure_reporting,
    get_reporting_config,
    hub_reporter,
    hub_tags,
)


@click.command()
# @click.option('--token', is_flag=True, default=False,
#        help='Enter authentication tocken from {}'.format(config.GET_TOKEN_REST_SUFFIX))
@click.option("--username", "-u", default=None, help="Your Activeloop AI username")
@click.option("--password", "-p", default=None, help="Your Activeloop AI password")
def login(username, password):
    """Log in to Activeloop AI"""
    login_fn(username, password)


@click.command()
def logout():
    """Log out of Activeloop AI"""
    TokenManager.purge_token()


@click.command()
@click.option("--username", "-u", default=None, help="Your Activeloop AI username")
@click.option("--email", "-e", default=None, help="Your email")
@click.option("--password", "-p", default=None, help="Your Activeloop AI password")
def register(username, email, password):
    """Register at Activeloop AI"""
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
    consent_message = textwrap.dedent(
        """
        Privacy policy:
        We collect basic system information and crash reports so that we can keep
        improving your experience using Hub to work with your data.

        You can find out more by reading our privacy policy:
            https://www.activeloop.ai/privacy/

        If you would like to opt out of reporting crashes and system information,
        run the following command:
            $ hub reporting --off
        """
    )
    logger.info(consent_message)
    configure_reporting(True, username=username)


@click.command()
@click.option("--on/--off", help="Turn crash report on/off")
def reporting(on):
    """
    Enable or disable sending crash reports to Activeloop AI.
    """
    report = Report(
        title="Consent change",
        tags=hub_reporter.system_tags() + hub_tags,
        content="Consent? `{}`".format(on),
    )
    hub_reporter.publish(report)
    configure_reporting(on)


def login_fn(username, password):
    """Log in to Activeloop AI"""
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
            "Please log in using Activeloop credentials. You can register at https://app.activeloop.ai "
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
    reporting_config = get_reporting_config()
    if reporting_config.get("username") != username:
        configure_reporting(True, username=username)
