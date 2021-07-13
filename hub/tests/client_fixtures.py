from hub.constants import HUB_CLOUD_OPT
from hub.tests.common import is_opt_true
import os
import pytest
from hub.client.client import HubBackendClient
import hub


@pytest.fixture(scope="session")
def hub_testing_credentials(request):
    if not is_opt_true(request, HUB_CLOUD_OPT):
        pytest.skip()

    hub.client.config.USE_DEV_ENVIRONMENT = True

    username = "testingacc"
    password = os.getenv("ACTIVELOOP_HUB_PASSWORD")
    return username, password


@pytest.fixture(scope="session")
def hub_testing_token(hub_testing_credentials):
    username, password = hub_testing_credentials

    # TODO: use DEV environment
    client = HubBackendClient()
    token = client.request_auth_token(username, password)
    return token
