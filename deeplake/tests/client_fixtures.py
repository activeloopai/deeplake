from deeplake.constants import (
    ENV_HUB_DEV_MANAGED_CREDS_KEY,
    AZURE_OPT,
    HUB_CLOUD_OPT,
    ENV_AZURE_CLIENT_ID,
    ENV_AZURE_CLIENT_SECRET,
    ENV_AZURE_TENANT_ID,
    ENV_HUB_DEV_USERNAME,
    ENV_HUB_DEV_TOKEN,
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
        pytest.skip(f"{HUB_CLOUD_OPT} flag not set")

    if not (USE_LOCAL_HOST or USE_DEV_ENVIRONMENT or USE_STAGING_ENVIRONMENT):
        warn(
            "Running deeplake cloud tests without setting USE_LOCAL_HOST, USE_DEV_ENVIRONMENT or USE_STAGING_ENVIRONMENT is not recommended."
        )

    username = os.getenv(ENV_HUB_DEV_USERNAME)

    assert (
        username is not None
    ), f"Deep Lake dev username was not found in the environment variable '{ENV_HUB_DEV_USERNAME}'. This is necessary for testing deeplake cloud datasets."

    return username, None


@pytest.fixture(scope="session")
def hub_cloud_dev_token(hub_cloud_dev_credentials):
    token = os.getenv(ENV_HUB_DEV_TOKEN)

    assert (
        token is not None
    ), f"Deep Lake dev token was not found in the environment variable '{ENV_HUB_DEV_TOKEN}'. This is necessary for testing deeplake cloud datasets."

    return token


@pytest.fixture(scope="session")
def hub_kaggle_credentials(request):
    if not is_opt_true(request, KAGGLE_OPT):
        pytest.skip(f"{KAGGLE_OPT} flag not set")

    username = os.getenv(ENV_KAGGLE_USERNAME)
    key = os.getenv(ENV_KAGGLE_KEY)

    assert (
        key is not None
    ), f"Kaggle credentials were not found in environment variable. This is necessary for testing kaggle ingestion datasets."

    return username, key


@pytest.fixture(scope="session")
def hub_cloud_dev_managed_creds_key(request):
    if not is_opt_true(request, HUB_CLOUD_OPT):
        pytest.skip(f"{HUB_CLOUD_OPT} flag not set")

    creds_key = os.getenv(ENV_HUB_DEV_MANAGED_CREDS_KEY, "aws_creds")
    return creds_key


@pytest.fixture(scope="session")
def azure_creds_key(request):
    if not is_opt_true(
        request,
        AZURE_OPT,
    ):
        pytest.skip(f"{AZURE_OPT} flag not set")

    creds_key = {
        "azure_client_id": os.getenv(ENV_AZURE_CLIENT_ID),
        "azure_tenant_id": os.getenv(ENV_AZURE_TENANT_ID),
        "azure_client_secret": os.getenv(ENV_AZURE_CLIENT_SECRET),
    }
    return creds_key
