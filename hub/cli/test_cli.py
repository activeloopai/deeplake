import os
import pytest
from click.testing import CliRunner

import hub
from hub.cli.auth import login, logout
from hub.cli.list_datasets import list_my_datasets, list_public_datasets


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
    ds = hub.dataset("hub://testingacc/test_list")

    res = runner.invoke(list_my_datasets)
    assert res.exit_code == 0
    assert "testingacc/test_list" in res.output

    res = runner.invoke(list_my_datasets, "--workspace activeloop")
    assert res.exit_code == 1

    res = runner.invoke(list_public_datasets)
    assert res.exit_code == 0
    assert "testingacc/test_list" in res.output

    res = runner.invoke(list_public_datasets, "--workspace activeloop")
    assert res.exit_code == 0
    assert "testingacc/test_list" not in res.output
    ds.delete()
