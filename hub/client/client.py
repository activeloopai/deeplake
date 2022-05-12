import hub
import requests
from typing import Optional
from hub.util.exceptions import LoginException, InvalidPasswordException
from hub.client.utils import check_response_status, write_token, read_token
from hub.client.config import (
    HUB_REST_ENDPOINT,
    HUB_REST_ENDPOINT_LOCAL,
    HUB_REST_ENDPOINT_DEV,
    GET_TOKEN_SUFFIX,
    HUB_REST_ENDPOINT_STAGING,
    REGISTER_USER_SUFFIX,
    DEFAULT_REQUEST_TIMEOUT,
    GET_DATASET_CREDENTIALS_SUFFIX,
    CREATE_DATASET_SUFFIX,
    DATASET_SUFFIX,
    LIST_DATASETS,
    GET_USER_PROFILE,
    SEND_EVENT_SUFFIX,
    UPDATE_SUFFIX,
    GET_PRESIGNED_URL_SUFFIX,
)
from hub.client.log import logger

# for these codes, we will retry requests upto 3 times
retry_status_codes = {502}


class HubBackendClient:
    """Communicates with Activeloop Backend"""

    def __init__(self, token: Optional[str] = None):
        self.version = hub.__version__
        self.auth_header = None
        if token is None:
            self.token = self.get_token()
        else:
            self.token = token
        self.auth_header = f"Bearer {self.token}"

    def get_token(self):
        """Returns a token"""
        token = read_token()
        if token is None:
            token = self.request_auth_token(username="public", password="")
            write_token(token)

        return token

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

        Raises:
            InvalidPasswordException: `password` cannot be `None` inside `json`.

        Returns:
            requests.Response: The response received from the server.
        """
        params = params or {}
        data = data or None
        files = files or None
        json = json or None
        endpoint = endpoint or self.endpoint()
        endpoint = endpoint.strip("/")
        relative_url = relative_url.strip("/")
        request_url = f"{endpoint}/{relative_url}"
        headers = headers or {}
        headers["hub-cli-version"] = self.version
        headers["Authorization"] = self.auth_header

        # clearer error than `ServerUnderMaintenence`
        if json is not None and "password" in json and json["password"] is None:
            # do NOT pass in the password here. `None` is explicitly typed.
            raise InvalidPasswordException("Password cannot be `None`.")

        status_code = None
        tries = 0
        while status_code is None or (status_code in retry_status_codes and tries < 3):
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
            status_code = response.status_code
            tries += 1
        check_response_status(response)
        return response

    def endpoint(self):
        if hub.client.config.USE_LOCAL_HOST:
            return HUB_REST_ENDPOINT_LOCAL
        if hub.client.config.USE_DEV_ENVIRONMENT:
            return HUB_REST_ENDPOINT_DEV
        if hub.client.config.USE_STAGING_ENVIRONMENT:
            return HUB_REST_ENDPOINT_STAGING

        return HUB_REST_ENDPOINT

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
        response = self.request("POST", GET_TOKEN_SUFFIX, json=json)

        try:
            token_dict = response.json()
            token = token_dict["token"]
        except Exception:
            raise LoginException()
        return token

    def send_register_request(self, username: str, email: str, password: str):
        """Sends a request to backend to register a new user.

        Args:
            username (str): The Activeloop username to create account for.
            email (str): The email id to link with the Activeloop account.
            password (str): The new password of the account. Should be atleast 6 characters long.
        """

        json = {"username": username, "email": email, "password": password}
        self.request("POST", REGISTER_USER_SUFFIX, json=json)

    def get_dataset_credentials(
        self, org_id: str, ds_name: str, mode: Optional[str] = None
    ):
        """Retrieves temporary 12 hour credentials for the required dataset from the backend.

        Args:
            org_id (str): The name of the user/organization to which the dataset belongs.
            ds_name (str): The name of the dataset being accessed.
            mode (str, optional): The mode in which the user has requested to open the dataset.
                If not provided, the backend will set mode to 'a' if user has write permission, else 'r'.

        Returns:
            tuple: containing full url to dataset, credentials, mode and expiration time respectively.
        """
        relative_url = GET_DATASET_CREDENTIALS_SUFFIX.format(org_id, ds_name)
        response = self.request(
            "GET",
            relative_url,
            endpoint=self.endpoint(),
            params={"mode": mode},
        ).json()
        full_url = response.get("path")
        creds = response["creds"]
        mode = response["mode"]
        expiration = creds["expiration"]
        return full_url, creds, mode, expiration

    def send_event(self, event_json: dict):
        """Sends an event to the backend.

        Args:
            event_json (dict): The event to be sent.
        """
        self.request("POST", SEND_EVENT_SUFFIX, json=event_json)

    def create_dataset_entry(self, username, dataset_name, meta, public=True):
        tag = f"{username}/{dataset_name}"
        repo = f"protected/{username}"

        response = self.request(
            "POST",
            CREATE_DATASET_SUFFIX,
            json={
                "tag": tag,
                "repository": repo,
                "public": public,
                "rewrite": True,
                "meta": meta,
            },
            endpoint=self.endpoint(),
        )

        if response.status_code == 200:
            logger.info("Your Hub dataset has been successfully created!")
            if public is False:
                logger.info("The dataset is private so make sure you are logged in!")

    def delete_dataset_entry(self, username, dataset_name):
        tag = f"{username}/{dataset_name}"
        suffix = f"{DATASET_SUFFIX}/{tag}"
        self.request(
            "DELETE",
            suffix,
            endpoint=self.endpoint(),
        ).json()

    def rename_dataset_entry(self, username, old_name, new_name):
        suffix = UPDATE_SUFFIX.format(username, old_name)
        self.request(
            "PUT", suffix, endpoint=self.endpoint(), json={"basename": new_name}
        )

    def get_user_organizations(self):
        """Get list of user organizations from the backend. If user is not logged in, returns ['public'].

        Returns:
            list: user/organization names
        """
        response = self.request(
            "GET", GET_USER_PROFILE, endpoint=self.endpoint()
        ).json()
        return response["organizations"]

    def get_workspace_datasets(
        self, workspace: str, suffix_public: str, suffix_user: str
    ):
        organizations = self.get_user_organizations()
        if workspace in organizations:
            response = self.request(
                "GET",
                suffix_user,
                endpoint=self.endpoint(),
                params={"organization": workspace},
            ).json()
        else:
            print(
                f'You are not a member of organization "{workspace}". List of accessible datasets from "{workspace}": ',
            )
            response = self.request(
                "GET",
                suffix_public,
                endpoint=self.endpoint(),
                params={"organization": workspace},
            ).json()
        return response

    def get_datasets(self, workspace: str):
        suffix_public = LIST_DATASETS.format("public")
        suffix_user = LIST_DATASETS.format("all")
        if workspace:
            res_datasets = self.get_workspace_datasets(
                workspace, suffix_public, suffix_user
            )
        else:
            public_datasets = self.request(
                "GET",
                suffix_public,
                endpoint=self.endpoint(),
            ).json()
            user_datasets = self.request(
                "GET",
                suffix_user,
                endpoint=self.endpoint(),
            ).json()
            res_datasets = public_datasets + user_datasets
        return [ds["_id"] for ds in res_datasets]

    def update_privacy(self, username: str, dataset_name: str, public: bool):
        suffix = UPDATE_SUFFIX.format(username, dataset_name)
        self.request("PUT", suffix, endpoint=self.endpoint(), json={"public": public})

    def get_presigned_url(self, org_id, ds_id, chunk_path, expiration=3600):
        relative_url = GET_PRESIGNED_URL_SUFFIX.format(org_id, ds_id)
        response = self.request(
            "GET",
            relative_url,
            endpoint=self.endpoint(),
            params={"chunk_path": chunk_path, "expiration": expiration},
        ).json()
        presigned_url = response["data"]
        return presigned_url
