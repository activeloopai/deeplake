import os

from deeplake.client.auth.auth_context import AuthContext, AuthProviderType
from deeplake.client.auth.activeloop import ActiveLoopAuthContext
from deeplake.client.auth.azure import AzureAuthContext
from deeplake.client.config import DEEPLAKE_AUTH_PROVIDER


def initialize_auth_context(*args, **kwargs) -> AuthContext:
    if (
        os.environ.get(DEEPLAKE_AUTH_PROVIDER, "").lower()
        == AuthProviderType.AZURE.value.lower()
    ):
        return AzureAuthContext()

    return ActiveLoopAuthContext(*args, **kwargs)


__all__ = ["initialize_auth_context"]
