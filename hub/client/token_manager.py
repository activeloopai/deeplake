import os
from pathlib import Path
from hub import config
from hub.log import logger


class TokenManager:
    """ manages access tokens """

    @classmethod
    def is_authenticated(cls):
        return os.path.getsize(config.TOKEN_FILE_PATH) > 10

    @classmethod
    def set_token(cls, token):
        logger.debug(f"Putting the key {token} into {config.TOKEN_FILE_PATH}.")
        path = Path(config.TOKEN_FILE_PATH)
        os.makedirs(path.parent, exist_ok=True)
        with open(config.TOKEN_FILE_PATH, "w") as f:
            f.write(token)

    @classmethod
    def get_token(cls):
        logger.debug("Getting token...")
        if not os.path.exists(config.TOKEN_FILE_PATH):
            return None
        with open(config.TOKEN_FILE_PATH) as f:
            token = f.read()
        logger.debug(f"Got the key {token} from {config.TOKEN_FILE_PATH}.")
        return token

    @classmethod
    def get_auth_header(cls):
        logger.debug("Constructing auth header...")
        token = cls.get_token()
        if token:
            return f"Bearer {token}"
        return None

    @classmethod
    def purge_token(cls):
        logger.debug("Purging token...")
        if os.path.isfile(config.TOKEN_FILE_PATH):
            os.remove(config.TOKEN_FILE_PATH)
