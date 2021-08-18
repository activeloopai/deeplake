import os
import pytest
from click.testing import CliRunner

import hub
from hub.cli.auth import login, logout
from hub.cli.list_datasets import list_datasets
from hub.tests.common import SESSION_ID


def test_cli_auth(hub_cloud_dev_credentials):
    username, password = hub_cloud_dev_credentials

    runner = CliRunner()

    result = runner.invoke(login, f"-u {username} -p {password}")
    assert result.exit_code == 0
    assert result.output == "Successfully logged in to Activeloop.\n"

    result = runner.invoke(logout)
    assert result.exit_code == 0
    assert result.output == "Logged out of Activeloop.\n"


def test_get_datasets(hub_cloud_dev_credentials):
    runner = CliRunner()
    username, password = hub_cloud_dev_credentials

    runner.invoke(login, f"-u {username} -p {password}")
    ds1 = hub.dataset(f"hub://testingacc/test_list_{SESSION_ID}")

    res = runner.invoke(list_datasets)
    assert res.exit_code == 0
    assert f"testingacc/test_list_{SESSION_ID}" in res.output

    res = runner.invoke(list_datasets, "--workspace activeloop")
    assert len(res.output.split("\n")) > 0

    ds2 = hub.dataset(f"hub://testingacc/test_list_private_{SESSION_ID}", public=False)
    res = runner.invoke(logout)
    assert res.output == "Logged out of Activeloop.\n"
    res = runner.invoke(list_datasets)
    assert f"testingacc/test_list_private_{SESSION_ID}" not in res.output

    ds1.delete()
    ds2.delete()
