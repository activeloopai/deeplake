from hub.client.config import HUB_REST_ENDPOINT
import os

import pytest
from click.testing import CliRunner

from hub.cli.auth import login, logout


def test_cli_auth(hub_cloud_dev_credentials):
    username, password = hub_cloud_dev_credentials

    runner = CliRunner()

    result = runner.invoke(login, f"-u {username} -p {password}")
    assert result.exit_code == 0
    assert result.output == "Successfully logged in to Activeloop.\n"

    result = runner.invoke(logout)
    assert result.exit_code == 0
    assert result.output == "Logged out of Activeloop.\n"
