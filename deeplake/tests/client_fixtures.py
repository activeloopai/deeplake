from deeplake.constants import (
    ENV_HUB_DEV_MANAGED_CREDS_KEY,
    HUB_CLOUD_OPT,
    ENV_HUB_DEV_USERNAME,
    ENV_HUB_DEV_PASSWORD,
    ENV_KAGGLE_USERNAME,
    ENV_KAGGLE_KEY,
    KAGGLE_OPT,
)
from deeplake.tests.common import is_opt_true
import os
import pytest
from deeplake.client.client import DeepLakeBackendClient
from deeplake.client.config import (
    USE_LOCAL_HOST,
    USE_DEV_ENVIRONMENT,
    USE_STAGING_ENVIRONMENT,
)

from warnings import warn


@pytest.fixture(scope="session")
def hub_cloud_dev_credentials(request):
    if not is_opt_true(request, HUB_CLOUD_OPT):
        pytest.skip()

    if not (USE_LOCAL_HOST or USE_DEV_ENVIRONMENT or USE_STAGING_ENVIRONMENT):
        warn(
            "Running deeplake cloud tests without setting USE_LOCAL_HOST, USE_DEV_ENVIRONMENT or USE_STAGING_ENVIRONMENT is not recommended."
        )

    username = os.getenv(ENV_HUB_DEV_USERNAME)
    password = os.getenv(ENV_HUB_DEV_PASSWORD)

    assert (
        username is not None
    ), f"Hub dev username was not found in the environment variable '{ENV_HUB_DEV_USERNAME}'. This is necessary for testing deeplake cloud datasets."
    assert (
        password is not None
    ), f"Hub dev password was not found in the environment variable '{ENV_HUB_DEV_PASSWORD}'. This is necessary for testing deeplake cloud datasets."

    return username, password


@pytest.fixture(scope="session")
def hub_cloud_dev_token(hub_cloud_dev_credentials):
    username, password = hub_cloud_dev_credentials

    client = DeepLakeBackendClient()
    token = client.request_auth_token(username, password)
    return token


@pytest.fixture(scope="session")
def hub_dev_token():
    token = "eyJhbGciOiJIUzUxMiIsImlhdCI6MTY1NjA3NjM0NywiZXhwIjo0ODA5Njc2MzQ3fQ.eyJpZCI6ImFkaWxraGFuIn0.SGTPTiVxE0YF4PY7aEt_9jtO9mQFrUHdq1tQIER_oh3cwjzLvsYvmWUQ32LPJu6axwgFfC7B-bohcYHu8iHAlw"
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


@pytest.fixture(scope="session")
def hub_cloud_dev_managed_creds_key(request):
    if not is_opt_true(request, HUB_CLOUD_OPT):
        pytest.skip()

    creds_key = os.getenv(ENV_HUB_DEV_MANAGED_CREDS_KEY, "deeplake_tests")
    return creds_key
