import os
from typing import Optional

from deeplake.client.auth.auth_context import AuthContext, AuthProviderType
from deeplake.client.config import DEEPLAKE_AUTH_TOKEN


class ActiveLoopAuthContext(AuthContext):
    def __init__(self, token: Optional[str] = None):
        self.token = token

    def get_token(self) -> Optional[str]:
        if self.token is None:
            self.authenticate()

        return self.token

    def is_public_user(self) -> bool:
        token = self.get_token()
        if token is None:
            return False

        return token.startswith("PUBLIC_TOKEN_")

    def authenticate(self) -> None:
        self.token = (
            self.token
            or os.environ.get(DEEPLAKE_AUTH_TOKEN)
            or "PUBLIC_TOKEN_" + ("_" * 150)
        )

    def get_provider_type(self) -> AuthProviderType:
        return AuthProviderType.ACTIVELOOP
