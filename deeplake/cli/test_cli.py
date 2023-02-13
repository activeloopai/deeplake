import os
import pytest
from click.testing import CliRunner

import deeplake
from deeplake.cli.auth import login, logout
from deeplake.cli.list_datasets import list_datasets
from deeplake.tests.common import SESSION_ID


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


@pytest.mark.parametrize("method", ["creds", "token"])
def test_get_datasets(hub_cloud_dev_credentials, hub_cloud_dev_token, method):
    runner = CliRunner()
    username, password = hub_cloud_dev_credentials

    if method == "creds":
        runner.invoke(login, f"-u {username} -p {password}")
    elif method == "token":
        runner.invoke(login, f"-t {hub_cloud_dev_token}")
        
    ds1 = deeplake.dataset(f"hub://{username}/test_list_{SESSION_ID}")

    res = runner.invoke(list_datasets)
    assert res.exit_code == 0
    assert f"{username}/test_list_{SESSION_ID}" in res.output

    res = runner.invoke(list_datasets, "--workspace activeloop")
    assert len(res.output.split("\n")) > 0

    ds2 = deeplake.dataset(
        f"hub://{username}/test_list_private_{SESSION_ID}", public=False
    )
    res = runner.invoke(logout)
    assert res.output == "Logged out of Activeloop.\n"
    res = runner.invoke(list_datasets)
    assert f"{username}/test_list_private_{SESSION_ID}" not in res.output

    ds1.delete()
    ds2.delete()
