import sys
from typing import Optional

import requests

from hub import __version__
from hub.client.config import (
    HUB_REST_ENDPOINT,
    GET_TOKEN_SUFFIX,
    REGISTER_USER_SUFFIX,
    DEFAULT_REQUEST_TIMEOUT,
)
from hub.client.utils import get_auth_header, check_response_status
from hub.util.exceptions import LoginException


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
        timeout: Optional[int] = DEFAULT_REQUEST_TIMEOUT,
    ):
        """Sends a request to the backend.

        Args:
            method (str): The method for sending the request. Should be one of 'GET', 'OPTIONS', 'HEAD', 'POST', 'PUT',
                'PATCH', or 'DELETE'.
            relative_url (str): The suffix to be appended to the end of the endpoint url.
            endpoint(str, optional): The endpoint to send the request to.
            params (dict, optional): Dictionary to send in the query string for the request.
            data (dict, optional): Dictionary to send in the body of the request.
            files (dict, optional): Dictionary of 'name': file-like-objects (or {'name': file-tuple}) for multipart
                encoding upload.
                file-tuple can be a 2-tuple (filename, fileobj), 3-tuple (filename, fileobj, content_type)
                or a 4-tuple (filename, fileobj, content_type, custom_headers), where 'content-type' is a string
                defining the content type of the given file and 'custom_headers' a dict-like object containing
                additional headers to add for the file.
            json (dict, optional): A JSON serializable Python object to send in the body of the request.
            headers (dict, optional): Dictionary of HTTP Headers to send with the request.
            timeout (float,optional): How many seconds to wait for the server to send data before giving up.

        Returns:
            requests.Response: The response received from the server.
        """
        params = params or {}
        data = data or {}
        files = files or {}
        json = json or {}
        endpoint = endpoint or HUB_REST_ENDPOINT
        endpoint = endpoint.strip("/")
        relative_url = relative_url.strip("/")
        request_url = f"{endpoint}/{relative_url}"
        headers = headers or {}
        headers["hub-cli-version"] = __version__
        self.auth_header = self.auth_header or get_auth_header()
        headers["Authorization"] = self.auth_header
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

    def request_auth_token(self, username: str, password: str):
        """Sends a request to backend to retrieve auth token.

        Args:
            username (str): The Activeloop username to request token for.
            password (str): The password of the account.

        Returns:
            string: The auth token corresponding to the accound.

        Raises:
            LoginException: If there is an issue retrieving the auth token.
        """
        json = {"username": username, "password": password}
        response = self.request("GET", GET_TOKEN_SUFFIX, json=json)

        try:
            token_dict = response.json()
            token = token_dict["token"]
        except Exception:
            raise LoginException()
        return token

    def send_register_request(self, username: str, email: str, password: str):
        """Sends a request to backend to register a new

        Args:
            username (str): The Activeloop username to create account for.
            email (str): The email id to link with the Activeloop account.
            password (str): The new password of the account. Should be atleast 6 characters long.
        """

        json = {"username": username, "email": email, "password": password}
        self.request("POST", REGISTER_USER_SUFFIX, json=json)

    def get_dataset_credentials(self, org_id: str, ds_name: str, mode: str):
        # waiting for AL-942
        pass
