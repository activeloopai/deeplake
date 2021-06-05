import os

import pytest
from click.testing import CliRunner

from hub.cli.auth import login, logout
from hub.client.utils import has_hub_testing_creds


@pytest.mark.skipif(not has_hub_testing_creds(), reason="requires hub credentials")
def test_cli_auth():
    runner = CliRunner()

    username = "testingacc"
    password = os.getenv("ACTIVELOOP_HUB_PASSWORD")
    result = runner.invoke(login, f"-u {username} -p {password}")
    assert result.exit_code == 0
    assert (
        result.output
        == "Log in using Activeloop credentials.\nIf you don't have an account register by using 'hub register' "
        "command or by going to https://app.activeloop.ai/register.\n\nSuccessfully logged in to Hub.\n"
    )

    result = runner.invoke(logout)
    assert result.exit_code == 0
    assert result.output == "Logged out of Hub.\n"
