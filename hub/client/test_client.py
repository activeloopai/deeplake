import pytest

from hub.client.client import HubBackendClient
from hub.client.utils import (
    write_token,
    read_token,
    remove_token,
)


def test_client_requests(hub_cloud_dev_credentials):
    username, password = hub_cloud_dev_credentials

    hub_client = HubBackendClient()
    hub_client.request_auth_token(username, password)
    with pytest.raises(Exception):
        # request will fail as username already exists
        hub_client.send_register_request("activeloop", "abc@d.com", "notactualpassword")


def test_client_utils():
    write_token("abcdefgh")
    assert read_token() == "abcdefgh"
    remove_token()
    assert read_token() is None
