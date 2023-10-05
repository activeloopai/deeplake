import deeplake
import requests
import textwrap
from typing import Any, Optional, Dict, List, Union
from deeplake.util.exceptions import (
    AgreementNotAcceptedError,
    AuthorizationException,
    LoginException,
    InvalidPasswordException,
    ManagedCredentialsNotFoundError,
    NotLoggedInAgreementError,
    ResourceNotFoundException,
    InvalidTokenException,
    UserNotLoggedInException,
    TokenPermissionError,
)
from deeplake.client.utils import (
    check_response_status,
    write_token,
    read_token,
    remove_token,
    JobResponseStatusSchema,
)
from deeplake.client.config import (
    ACCEPT_AGREEMENTS_SUFFIX,
    REJECT_AGREEMENTS_SUFFIX,
    GET_MANAGED_CREDS_SUFFIX,
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
    GET_USER_PROFILE,
    SEND_EVENT_SUFFIX,
    UPDATE_SUFFIX,
    GET_PRESIGNED_URL_SUFFIX,
    CONNECT_DATASET_SUFFIX,
    REMOTE_QUERY_SUFFIX,
    ORG_PERMISSION_SUFFIX,
)
from deeplake.client.log import logger
import jwt  # should add it to requirements.txt

# for these codes, we will retry requests upto 3 times
retry_status_codes = {502}


class DeepLakeBackendClient:
    """Communicates with Activeloop Backend"""

    def __init__(self, token: Optional[str] = None):
        from deeplake.util.bugout_reporter import (
            save_reporting_config,
            get_reporting_config,
            set_username,
        )

        self.version = deeplake.__version__
        self._token_from_env = False
        self.auth_header = None
        self.token = token or self.get_token()
        self.auth_header = f"Bearer {self.token}"

        # remove public token, otherwise env var will be ignored
        # we can remove this after a while
        orgs = self.get_user_organizations()
        if orgs == ["public"]:
            remove_token()
            self.token = token or self.get_token()
            self.auth_header = f"Bearer {self.token}"
        if self._token_from_env:
            username = self.get_user_profile()["name"]
            if get_reporting_config().get("username") != username:
                save_reporting_config(True, username=username)
                set_username(username)

    def get_token(self):
        """Returns a token"""
        self._token_from_env = False
        token = read_token(from_env=False)
        if token is None:
            token = read_token(from_env=True)
            if token is None:
                token = self.request_auth_token(username="public", password="")
            else:
                self._token_from_env = True
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
        if deeplake.client.config.USE_LOCAL_HOST:
            return HUB_REST_ENDPOINT_LOCAL
        if deeplake.client.config.USE_DEV_ENVIRONMENT:
            return HUB_REST_ENDPOINT_DEV
        if deeplake.client.config.USE_STAGING_ENVIRONMENT:
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
            UserNotLoggedInException: if user is not authorised
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
        self,
        org_id: str,
        ds_name: str,
        mode: Optional[str] = None,
        db_engine: Optional[dict] = None,
        no_cache: bool = False,
    ):
        """Retrieves temporary 12 hour credentials for the required dataset from the backend.

        Args:
            org_id (str): The name of the user/organization to which the dataset belongs.
            ds_name (str): The name of the dataset being accessed.
            mode (str, optional): The mode in which the user has requested to open the dataset.
                If not provided, the backend will set mode to 'a' if user has write permission, else 'r'.
            db_engine (dict, optional): The database engine args to use for the dataset.
            no_cache (bool): If True, cached creds are ignored and new creds are returned. Default False.

        Returns:
            tuple: containing full url to dataset, credentials, mode and expiration time respectively.

        Raises:
            UserNotLoggedInException: When user is not logged in
            InvalidTokenException: If the specified token is invalid
            TokenPermissionError: when there are permission or other errors related to token
            AgreementNotAcceptedError: when user has not accepted the agreement
            NotLoggedInAgreementError: when user is not logged in and dataset has agreement which needs to be signed
        """
        import json

        db_engine = db_engine or {}
        relative_url = GET_DATASET_CREDENTIALS_SUFFIX.format(org_id, ds_name)
        try:
            response = self.request(
                "GET",
                relative_url,
                endpoint=self.endpoint(),
                params={
                    "mode": mode,
                    "no_cache": no_cache,
                    "db_engine": json.dumps(db_engine),
                },
            ).json()
        except Exception as e:
            if isinstance(e, AuthorizationException):
                authorization_exception_prompt = "You don't have permission"
                response_data = e.response.json()
                code = response_data.get("code")
                if code == 1:
                    agreements = response_data["agreements"]
                    agreements = [agreement["text"] for agreement in agreements]
                    raise AgreementNotAcceptedError(agreements) from e
                elif code == 2:
                    raise NotLoggedInAgreementError from e
                else:
                    try:
                        decoded_token = jwt.decode(
                            self.token, options={"verify_signature": False}
                        )
                    except Exception:
                        raise InvalidTokenException

                    if (
                        authorization_exception_prompt.lower()
                        in response_data["description"].lower()
                        and decoded_token["id"] == "public"
                    ):
                        raise UserNotLoggedInException()
                    raise TokenPermissionError()
            raise
        full_url = response.get("path")
        repository = response.get("repository")
        creds = response["creds"]
        mode = response["mode"]
        expiration = creds["expiration"] if creds else None
        return full_url, creds, mode, expiration, repository

    def send_event(self, event_json: dict):
        """Sends an event to the backend.

        Args:
            event_json (dict): The event to be sent.
        """
        self.request("POST", SEND_EVENT_SUFFIX, json=event_json)

    def create_dataset_entry(
        self, username, dataset_name, meta, public=True, repository=None
    ):
        tag = f"{username}/{dataset_name}"
        if repository is None:
            repository = f"protected/{username}"

        response = self.request(
            "POST",
            CREATE_DATASET_SUFFIX,
            json={
                "tag": tag,
                "public": public,
                "rewrite": True,
                "meta": meta,
                "repository": repository,
            },
            endpoint=self.endpoint(),
        )

        if response.status_code == 200:
            logger.info("Your Deep Lake dataset has been successfully created!")

    def get_managed_creds(self, org_id, creds_key):
        """Retrieves the managed credentials for the given org_id and creds_key.

        Args:
            org_id (str): The name of the user/organization to which the dataset belongs.
            creds_key (str): The key corresponding to the managed credentials.

        Returns:
            dict: The managed credentials.

        Raises:
            ManagedCredentialsNotFoundError: If the managed credentials do not exist for the given organization.
        """
        relative_url = GET_MANAGED_CREDS_SUFFIX.format(org_id)
        try:
            resp = self.request(
                "GET",
                relative_url,
                endpoint=self.endpoint(),
                params={"query": creds_key},
            ).json()
        except ResourceNotFoundException:
            raise ManagedCredentialsNotFoundError(org_id, creds_key) from None
        creds = resp["creds"]
        key_mapping = {
            "access_key": "aws_access_key_id",
            "secret_key": "aws_secret_access_key",
            "session_token": "aws_session_token",
            "token": "aws_session_token",
            "region": "aws_region",
        }
        final_creds = {}
        for key, value in creds.items():
            if key == "access_token":
                key = "Authorization"
                value = f"Bearer {value}"
            elif key in key_mapping:
                key = key_mapping[key]
            final_creds[key] = value
        return final_creds

    def delete_dataset_entry(self, username, dataset_name):
        tag = f"{username}/{dataset_name}"
        suffix = f"{DATASET_SUFFIX}/{tag}"
        self.request(
            "DELETE",
            suffix,
            endpoint=self.endpoint(),
        ).json()

    def accept_agreements(self, org_id, ds_name):
        """Accepts the agreements for the given org_id and ds_name.

        Args:
            org_id (str): The name of the user/organization to which the dataset belongs.
            ds_name (str): The name of the dataset being accessed.
        """
        relative_url = ACCEPT_AGREEMENTS_SUFFIX.format(org_id, ds_name)
        self.request(
            "POST",
            relative_url,
            endpoint=self.endpoint(),
        ).json()

    def reject_agreements(self, org_id, ds_name):
        """Rejects the agreements for the given org_id and ds_name.

        Args:
            org_id (str): The name of the user/organization to which the dataset belongs.
            ds_name (str): The name of the dataset being accessed.
        """
        relative_url = REJECT_AGREEMENTS_SUFFIX.format(org_id, ds_name)
        self.request(
            "POST",
            relative_url,
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

    def get_user_profile(self):
        response = self.request(
            "GET",
            "/api/user/profile",
            endpoint=self.endpoint(),
        )
        return response.json()

    def connect_dataset_entry(
        self,
        src_path: str,
        org_id: str,
        ds_name: Optional[str] = None,
        creds_key: Optional[str] = None,
    ) -> str:
        """Creates a new dataset entry that can be accessed with a hub path, but points to the original ``src_path``.

        Args:
            src_path (str): The path at which the source dataset resides.
            org_id (str): The organization into which the dataset entry is put and where the credentials are searched.
            ds_name (Optional[str]): Name of the dataset entry. Can be infered from the source path.
            creds_key (Optional[str]): Name of the managed credentials that will be used to access the source path.

        Returns:
            str: The id of the dataset entry that was created.
        """
        response = self.request(
            "POST",
            CONNECT_DATASET_SUFFIX,
            json={
                "src_path": src_path,
                "org_id": org_id,
                "ds_name": ds_name,
                "creds_key": creds_key,
            },
            endpoint=self.endpoint(),
        ).json()

        return response["generated_id"]

    def remote_query(
        self, org_id: str, ds_name: str, query_string: str
    ) -> Dict[str, Any]:
        """Queries a remote dataset.

        Args:
            org_id (str): The organization to which the dataset belongs.
            ds_name (str): The name of the dataset.
            query_string (str): The query string.

        Returns:
            Dict[str, Any]: The json response containing matching indicies and data from virtual tensors.
        """
        response = self.request(
            "POST",
            REMOTE_QUERY_SUFFIX.format(org_id, ds_name),
            json={"query": query_string},
            endpoint=self.endpoint(),
        ).json()

        return response

    def has_indra_org_permission(self, org_id: str) -> Dict[str, Any]:
        """Queries a remote dataset.

        Args:
            org_id (str): The organization to which the dataset belongs.

        Returns:
            Dict[str, Any]: The json response containing org permissions.
        """
        response = self.request(
            "GET",
            ORG_PERMISSION_SUFFIX.format(org_id),
            endpoint=self.endpoint(),
        ).json()

        return response


class DeepMemoryBackendClient(DeepLakeBackendClient):
    def __init__(self, token: Optional[str] = None):
        super().__init__(token=token)

    def deepmemory_is_available(self, org_id: str):
        """Checks if DeepMemory is available for the user.
        Args:
            org_id (str): The name of the user/organization to which the dataset belongs.
        Returns:
            bool: True if DeepMemory is available, False otherwise.
        """
        try:
            response = self.request(
                "GET",
                f"/api/organizations/{org_id}/features/deepmemory",
                endpoint=self.endpoint(),
            )
            return response.json()["available"]
        except Exception:
            return False

    def start_taining(
        self,
        corpus_path: str,
        queries_path: str,
    ) -> Dict[str, Any]:
        """Starts training of DeepMemory model.
        Args:
            corpus_path (str): The path to the corpus dataset.
            queries_path (str): The path to the queries dataset.
        Returns:
            Dict[str, Any]: The json response containing job_id.
        """
        response = self.request(
            method="POST",
            relative_url="/api/deepmemory/v1/train",
            json={"corpus_dataset": corpus_path, "query_dataset": queries_path},
        )
        check_response_status(response)
        return response.json()

    def cancel_job(self, job_id: str):
        """Cancels a job with job_id.
        Args:
            job_id (str): The job_id of the job to be cancelled.
        Returns:
            bool: True if job was cancelled successfully, False otherwise.
        """
        try:
            response = self.request(
                method="POST",
                relative_url=f"/api/deepmemory/v1/jobs/{job_id}/cancel",
            )
            check_response_status(response)
        except Exception as e:
            print(f"Job with job_id='{job_id}' was not cancelled!\n Error: {e}")
            return False
        print("Job cancelled successfully")
        return True

    def check_status(self, job_id: str, recall: str, improvement: str):
        """Checks status of a job with job_id.
        Args:
            job_id (str): The job_id of the job to be checked.
            recall (str): Current best top 10 recall
            importvement (str): Current best improvement over baseline
        Returns:
            Dict[str, Any]: The json response containing job status.
        """
        response = self.request(
            method="GET",
            relative_url=f"/api/deepmemory/v1/jobs/{job_id}/status",
        )
        check_response_status(response)
        response_status_schema = JobResponseStatusSchema(response=response.json())
        response_status_schema.print_status(job_id, recall, improvement)
        return response.json()

    def list_jobs(self, dataset_path: str):
        """Lists all jobs for a dataset.
        Args:
            dataset_path (str): The path to the dataset.
        Returns:
            Dict[str, Any]: The json response containing list of jobs.
        """
        dataset_id = dataset_path[6:]
        response = self.request(
            method="GET",
            relative_url=f"/api/deepmemory/v1/{dataset_id}/jobs",
        )
        check_response_status(response)
        return response.json()

    def delete_job(self, job_id: str):
        """Deletes a job with job_id.
        Args:
            job_id (str): The job_id of the job to be deleted.
        Returns:
            bool: True if job was deleted successfully, False otherwise.
        """
        try:
            response = self.request(
                method="DELETE",
                relative_url=f"/api/deepmemory/v1/jobs/{job_id}",
            )
            check_response_status(response)
            return True
        except Exception as e:
            print(f"Job with job_id='{job_id}' was not deleted!\n Error: {e}")
            return False
