import os
import pytest

from deeplake.client.auth import AuthProviderType, initialize_auth_context
from deeplake.client.auth.azure import AzureAuthContext
from deeplake.client.config import DEEPLAKE_AUTH_PROVIDER


def test_initialize_auth_context():
    context = initialize_auth_context(token="dummy")
    assert context.get_provider_type() == AuthProviderType.ACTIVELOOP

    context = initialize_auth_context()
    assert context.get_provider_type() == AuthProviderType.ACTIVELOOP

    os.environ[DEEPLAKE_AUTH_PROVIDER] = "dummy"
    context = initialize_auth_context()
    assert context.get_provider_type() == AuthProviderType.ACTIVELOOP

    os.environ[DEEPLAKE_AUTH_PROVIDER] = "azure"
    context = initialize_auth_context()
    assert isinstance(context, AzureAuthContext)
    assert context.get_provider_type() == AuthProviderType.AZURE

    os.environ[DEEPLAKE_AUTH_PROVIDER] = "AzUrE"
    context = initialize_auth_context()
    assert isinstance(context, AzureAuthContext)
    assert context.get_provider_type() == AuthProviderType.AZURE

    del os.environ[DEEPLAKE_AUTH_PROVIDER]


@pytest.mark.skip(
    reason="This test requires not having Azure credentials and fails in the CI environment."
)
def test_azure_auth_context_exceptions():
    azure_envs = [i for i in os.environ if i.startswith("AZURE_")]
    values = {i: os.environ[i] for i in azure_envs}
    context = AzureAuthContext()

    for key in azure_envs:
        del os.environ[key]

    with pytest.raises(Exception):
        context.authenticate()

    for key, value in values.items():
        os.environ[key] = value
