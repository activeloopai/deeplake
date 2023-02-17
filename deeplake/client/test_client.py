import pytest
from deeplake.cli.commands import login, logout
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


@pytest.mark.parametrize("method", ["creds", "token"])
def test_client_workspace_organizations(
    method, hub_cloud_dev_credentials, hub_cloud_dev_token
):
    username, password = hub_cloud_dev_credentials
    deeplake_client = DeepLakeBackendClient()

    runner = CliRunner()
    result = runner.invoke(logout)
    assert result.exit_code == 0

    assert deeplake_client.get_user_organizations() == ["public"]

    if method == "creds":
        runner.invoke(login, f"-u {username} -p {password}")
    elif method == "token":
        runner.invoke(login, f"-t {hub_cloud_dev_token}")

    deeplake_client = DeepLakeBackendClient()
    assert username in deeplake_client.get_user_organizations()

    runner.invoke(logout)
