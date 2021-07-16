from hub.constants import (
    HUB_CLOUD_DEV_USERNAME,
    HUB_CLOUD_OPT,
    HUB_DEV_PASSWORD_ENVIRONMENT_VARIABLE,
)
from hub.tests.common import is_opt_true
import os
import pytest
from hub.client.client import HubBackendClient


@pytest.fixture(scope="session")
def hub_cloud_dev_credentials(request):
    if not is_opt_true(request, HUB_CLOUD_OPT):
        pytest.skip()

    # TODO: use dev environment

    password = os.getenv(HUB_DEV_PASSWORD_ENVIRONMENT_VARIABLE)

    assert (
        password is not None
    ), f"Hub dev password was not found in the environment variable '{HUB_DEV_PASSWORD_ENVIRONMENT_VARIABLE}'. This is necessary for testing hub cloud datasets."

    return HUB_CLOUD_DEV_USERNAME, password


@pytest.fixture(scope="session")
def hub_cloud_dev_token(hub_cloud_dev_credentials):
    username, password = hub_cloud_dev_credentials

    client = HubBackendClient()
    token = client.request_auth_token(username, password)
    return token
