import os

from hub import config
from hub.log import logger


class TokenManager(object):
    """ manages access tokens """

    @classmethod
    def is_authenticated(cls):
        return os.path.getsize(config.TOKEN_FILE_PATH) > 10

    @classmethod
    def set_token(cls, token):
        logger.debug(
            "Putting the key {} into {}.".format(token, config.TOKEN_FILE_PATH)
        )
        with open(config.TOKEN_FILE_PATH, "w") as f:
            f.write(token)

    @classmethod
    def get_token(cls):
        logger.debug("Getting token...")
        with open(config.TOKEN_FILE_PATH, "r") as f:
            token = f.read()
        logger.debug("Got the key {} from {}.".format(token, config.TOKEN_FILE_PATH))
        return token

    @classmethod
    def get_auth_header(cls):
        logger.debug("Constructing auth header...")
        token = cls.get_token()
        auth_header = "Bearer {}".format(token)
        return auth_header

    @classmethod
    def purge_token(cls):
        logger.debug("Purging token...")
        if os.path.isfile(config.TOKEN_FILE_PATH):
            os.remove(config.TOKEN_FILE_PATH)
