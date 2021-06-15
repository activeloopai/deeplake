"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from hub_v1 import config
from hub_v1.log import logger
from hub_v1.client.base import HubHttpClient
from hub_v1.exceptions import HubException


class AuthClient(HubHttpClient):
    """
    Authentication through rest api
    """

    def __init__(self):
        super().__init__()

    def check_token(self, access_token):
        auth = f"Bearer {access_token}"
        response = self.request(
            "GET", config.CHECK_TOKEN_REST_SUFFIX, headers={"Authorization": auth}
        )

        try:
            response_dict = response.json()
            is_valid = response_dict["is_valid"]
        except Exception as e:
            logger.error(f"Exception occured while validating token: {e}.")
            raise HubException(
                "Error while validating the token. \
                                  Please try logging in using username ans password."
            )

        return is_valid

    def get_access_token(self, username, password):
        response = self.request(
            "GET",
            config.GET_TOKEN_REST_SUFFIX,
            json={"username": username, "password": password},
        )

        try:
            token_dict = response.json()
            token = token_dict["token"]
        except Exception as e:
            logger.error(f"Exception occured while getting token: {e}.")
            raise HubException(
                "Error while loggin in. \
                                  Please try logging in using access token."
            )
        return token

    def register(self, username, email, password):
        self.request(
            "POST",
            config.GET_REGISTER_SUFFIX,
            json={"username": username, "email": email, "password": password},
        )

        # try:
        #    token_dict = response.json()
        # except Exception as e:
        #    logger.error("Exception occured while registering token: {}.".format(e))
        #    raise HubException("Error while registering in. {e}")
