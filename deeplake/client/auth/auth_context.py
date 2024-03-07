from enum import Enum
from abc import ABC, abstractmethod


class AuthProviderType(Enum):
    ACTIVELOOP = "activeloop"
    AZURE = "azure"


class AuthContext(ABC):
    def get_auth_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.get_token()}",
            "X-Activeloop-Provider-Type": self.get_provider_type().value,
        }

    @abstractmethod
    def get_token(self) -> str:
        pass

    @abstractmethod
    def authenticate(self) -> None:
        """
        Try to authenticate using the necessary configuration.
        If the authentication fails, an `InvalidAuthContext` exception should be raised.
        """
        pass

    @abstractmethod
    def get_provider_type(self) -> AuthProviderType:
        pass
