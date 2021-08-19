from hub.constants import (
    HUB_CLOUD_DEV_USERNAME,
    HUB_CLOUD_OPT,
    ENV_HUB_DEV_PASSWORD,
    ENV_KAGGLE_USERNAME,
    ENV_KAGGLE_KEY,
    KAGGLE_OPT,
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

    password = os.getenv(ENV_HUB_DEV_PASSWORD)

    assert (
        password is not None
    ), f"Hub dev password was not found in the environment variable '{ENV_HUB_DEV_PASSWORD}'. This is necessary for testing hub cloud datasets."

    return HUB_CLOUD_DEV_USERNAME, password


@pytest.fixture(scope="session")
def hub_cloud_dev_token(hub_cloud_dev_credentials):
    username, password = hub_cloud_dev_credentials

    client = HubBackendClient()
    token = client.request_auth_token(username, password)
    return token


@pytest.fixture(scope="session")
def hub_kaggle_credentials(request):
    if not is_opt_true(request, KAGGLE_OPT):
        pytest.skip()

    username = os.getenv(ENV_KAGGLE_USERNAME)
    key = os.getenv(ENV_KAGGLE_KEY)

    assert (
        key is not None
    ), f"Kaggle credentials were not found in environment variable. This is necessary for testing kaggle ingestion datasets."

    return username, key
