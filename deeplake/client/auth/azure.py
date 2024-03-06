from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ClientAuthenticationError

from deeplake.client.auth.auth_context import AuthContext, AuthProviderType
from deeplake.util.exceptions import InvalidAuthContextError

class AzureAuthContext(AuthContext):
    def __init__(self):
        self.credential = DefaultAzureCredential()
        self.token = None

    def get_token(self) -> str:
        self.authenticate()

        return self.token

    def authenticate(self) -> None:
        try:
            response = self.credential.get_token("https://management.azure.com/.default")
            self.token = response.token
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