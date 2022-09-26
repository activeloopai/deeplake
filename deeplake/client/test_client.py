import subprocess
import pytest
from deeplake.cli.commands import login
from click.testing import CliRunner
from deeplake.client.client import DeepLakeBackendClient
from deeplake.client.utils import (
    write_token,
    read_token,
    remove_token,
)


def test_client_requests(hub_cloud_dev_credentials):
    username, password = hub_cloud_dev_credentials

    deeplake_client = DeepLakeBackendClient()
    deeplake_client.request_auth_token(username, password)
    with pytest.raises(Exception):
        # request will fail as username already exists
        deeplake_client.send_register_request(
            "activeloop", "abc@d.com", "notactualpassword"
        )


def test_client_utils():
    write_token("abcdefgh")
    assert read_token() == "abcdefgh"
    remove_token()
    assert read_token() is None


def test_client_workspace_organizations(hub_cloud_dev_credentials):
    username, password = hub_cloud_dev_credentials
    deeplake_client = DeepLakeBackendClient()

    assert deeplake_client.get_user_organizations() == ["public"]
    token = deeplake_client.request_auth_token(username, password)
    runner = CliRunner()
    runner.invoke(login, f"-u {username} -p {password}")
    deeplake_client = DeepLakeBackendClient()
    assert username in deeplake_client.get_user_organizations()
    assert "public" in deeplake_client.get_user_organizations()

    datasets = subprocess.check_output(
        ["activeloop", "list-datasets", "--workspace", "activeloop"]
    )
    assert "You are not a member of organization" in str(datasets)
    datasets = subprocess.check_output(
        ["activeloop", "list-datasets", "--workspace", "test"]
    )
    assert "You are not a member of organization" not in str(datasets)
