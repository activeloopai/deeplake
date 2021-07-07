from hub.client.config import HUB_REST_ENDPOINT
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
    assert result.output == "Successfully logged in to Activeloop.\n"

    result = runner.invoke(logout)
    assert result.exit_code == 0
    assert result.output == "Logged out of Activeloop.\n"
