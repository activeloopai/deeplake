from deeplake.util.exceptions import LoginException
from deeplake.cli.auth import login, logout
from click.testing import CliRunner

import pytest


@pytest.mark.parametrize("method", ["creds", "token"])
def test_cli_auth(hub_cloud_dev_credentials, hub_cloud_dev_token, method):
    username, password = hub_cloud_dev_credentials

    runner = CliRunner()

    if method == "creds":
        result = runner.invoke(login, f"-u {username} -p {password}")
    elif method == "token":
        result = runner.invoke(login, f"-t {hub_cloud_dev_token}")

    assert result.exit_code == 0
    assert result.output == "Successfully logged in to Activeloop.\n"

    result = runner.invoke(logout)
    assert result.exit_code == 0
    assert result.output == "Logged out of Activeloop.\n"


def test_bad_token():
    runner = CliRunner()

    result = runner.invoke(login, "-t abcd")
    assert isinstance(result.exception, LoginException)
