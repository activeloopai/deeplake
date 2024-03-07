import os
import pytest
from unittest.mock import Mock

from deeplake.client.auth import AuthProviderType, initialize_auth_context
from deeplake.client.auth.azure import AzureAuthContext
from deeplake.client.auth.activeloop import ActiveLoopAuthContext
from deeplake.client.config import DEEPLAKE_AUTH_PROVIDER
from deeplake.util.exceptions import InvalidAuthContextError


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


def test_azure_auth_context_exceptions():
    context = AzureAuthContext()
    
    with pytest.raises(InvalidAuthContextError):
        context.authenticate()

    context.credentials = Mock()
    context.credentials.get_token = Mock(return_value=None, side_effect=Exception)

    with pytest.raises(InvalidAuthContextError):
        context.authenticate()