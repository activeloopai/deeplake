import requests
import sys
from hub.client.config import (
    HUB_REST_ENDPOINT,
    GET_TOKEN_SUFFIX,
    REGISTER_USER_SUFFIX,
    DEFAULT_TIMEOUT,
)
from hub.client.utils import get_auth_header, check_response_status
from hub.util.exceptions import LoginException
from typing import Optional


class HubBackendClient:
    """Communicates with Activeloop Backend"""

    def __init__(self):
        self.auth_header = None

    def request(
        self,
        method: str,
        relative_url: str,
        endpoint: Optional[str] = None,
        params: Optional[dict] = None,
        data: Optional[dict] = None,
        files: Optional[dict] = None,
        json: Optional[dict] = None,
        headers: Optional[dict] = None,
        timeout: Optional[int] = DEFAULT_TIMEOUT,
    ):
        params = params or {}
        data = data or {}
        files = files or {}
        json = json or {}
        endpoint = endpoint or HUB_REST_ENDPOINT
        endpoint = endpoint.strip("/")
        relative_url = relative_url.strip("/")
        request_url = f"{endpoint}/{relative_url}"
        headers = headers or {}
        headers["hub-cli-version"] = "2.0"
        headers["Authorization"] = self.auth_header or get_auth_header()
        try:
            response = requests.request(
                method,
                request_url,
                params=params,
                data=data,
                json=json,
                headers=headers,
                files=files,
                timeout=timeout,
            )
        except requests.exceptions.ConnectionError as e:
            sys.exit("Connection error. Please retry or check your internet connection")
        except requests.exceptions.Timeout as e:
            sys.exit(
                "Connection timeout. Please retry or check your internet connection"
            )
        check_response_status(response)
        return response

    def request_access_token(self, username: str, password: str):
        json = {"username": username, "password": password}
        response = self.request("GET", GET_TOKEN_SUFFIX, json=json)

        try:
            token_dict = response.json()
            token = token_dict["token"]
        except Exception:
            raise LoginException()
        return token

    def send_register_request(self, username: str, email: str, password: str):
        json = {"username": username, "email": email, "password": password}
        self.request("POST", REGISTER_USER_SUFFIX, json=json)

    def get_dataset_credentials(self, org_id: str, ds_name: str, mode: str):
        # waiting for AL-942
        pass
