import os

import pytest

from hub.client.client import HubBackendClient
from hub.client.utils import (
    write_token,
    get_auth_header,
    remove_token,
    has_hub_testing_creds,
)


@pytest.mark.skipif(not has_hub_testing_creds(), reason="requires hub credentials")
def test_client_requests():
    username = "testingacc"
    password = os.getenv("ACTIVELOOP_HUB_PASSWORD")
    hub_client = HubBackendClient()

    hub_client.request_auth_token(username, password)
    with pytest.raises(Exception):
        # request will fail as username already exists
        hub_client.send_register_request("activeloop", "abc@d.com", "notactualpassword")


def test_client_utils():
    write_token("abcdefgh")
    assert get_auth_header() == "Bearer abcdefgh"
    remove_token()
    assert get_auth_header() is None
