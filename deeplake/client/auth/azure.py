import os
from datetime import datetime, timedelta
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
from azure.core.exceptions import ClientAuthenticationError

from deeplake.client.auth.auth_context import AuthContext, AuthProviderType
from deeplake.util.exceptions import InvalidAuthContextError

ID_TOKEN_CACHE_MINUTES = 5


class AzureAuthContext(AuthContext):
    def __init__(self):
        self.credential = self._get_azure_credential()
        self.token = None
        self._last_auth_time = None

    def _get_azure_credential(self):
        azure_keys = [i for i in os.environ if i.startswith("AZURE_")]
        if "AZURE_CLIENT_ID" in azure_keys and len(azure_keys) == 1:
            # Explicitly set client_id, to avoid any warnings coming from DefaultAzureCredential
            return ManagedIdentityCredential(
                client_id=os.environ.get("AZURE_CLIENT_ID")
            )

        return DefaultAzureCredential()

    def get_token(self) -> str:
        self.authenticate()

        return self.token

    def authenticate(self) -> None:
        if (
            self._last_auth_time is not None
            and datetime.now() - self._last_auth_time
            < timedelta(minutes=ID_TOKEN_CACHE_MINUTES)
        ):
            return

        try:
            response = self.credential.get_token(
                "https://management.azure.com/.default"
            )
            self.token = response.token
            self._last_auth_time = datetime.now()
        except ClientAuthenticationError as e:
            raise InvalidAuthContextError(
                f"Failed to authenticate with Azure. Please check your credentials. \n {e.message}",
            ) from e
        except Exception as e:
            raise InvalidAuthContextError(
                "Failed to authenticate with Azure. An unexpected error occured."
            ) from e

    def get_provider_type(self) -> AuthProviderType:
        return AuthProviderType.AZURE
