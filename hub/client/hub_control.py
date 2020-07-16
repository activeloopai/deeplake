import json
import os
import time
from hub import config
from hub.client.base import HubHttpClient

# import subprocess
# from subprocess import Popen, PIPE
# from hub.log import logger
# from hub.exceptions import HubException


class HubControlClient(HubHttpClient):
    """
    Controlling Hub through rest api
    """

    def __init__(self):
        super(HubControlClient, self).__init__()
        self.details = self.get_config()

    def get_credentials(self):
        r = self.request(
            "GET", config.GET_CREDENTIALS_SUFFIX, endpoint=config.HUB_REST_ENDPOINT,
        ).json()

        details = {
            "_id": r["_id"],
            "region": r["region"],
            "session_token": r["session_token"],
            "access_key": r["access_key"],
            "secret_key": r["secret_key"],
            "endpoint": r["endpoint"],
            "expiration": r["expiration"],
            "bucket": r["bucket"],
        }

        self.save_config(details)
        return details

    def get_config(self, reset=False):
        if not os.path.isfile(config.STORE_CONFIG_PATH):
            self.get_credentials()

        with open(config.STORE_CONFIG_PATH, "r") as file:
            details = file.readlines()
            details = json.loads("".join(details))

        if float(details["expiration"]) < time.time() or reset:
            details = self.get_credentials()
        return details

    def save_config(self, details):
        with open(config.STORE_CONFIG_PATH, "w") as file:
            file.writelines(json.dumps(details))
