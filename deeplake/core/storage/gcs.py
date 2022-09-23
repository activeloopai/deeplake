import datetime
import posixpath
import pickle
import json
import os
import tempfile
import time
from typing import Dict, Optional, Tuple, Union

try:
    from google.cloud import storage  # type: ignore
    from google.api_core import retry  # type: ignore
    from google.oauth2 import service_account  # type: ignore
    import google.auth as gauth  # type: ignore
    import google.auth.compute_engine  # type: ignore
    import google.auth.credentials  # type: ignore
    import google.auth.exceptions  # type: ignore
    from google_auth_oauthlib.flow import InstalledAppFlow  # type: ignore
    from google.api_core.exceptions import NotFound  # type: ignore

    _GOOGLE_PACKAGES_INSTALLED = True
except ImportError:
    _GOOGLE_PACKAGES_INSTALLED = False


from hub.core.storage.provider import StorageProvider
from hub.util.exceptions import (
    GCSDefaultCredsNotFoundError,
    RenameError,
    PathNotEmptyException,
)
from hub.client.client import HubBackendClient


def _remove_protocol_from_path(path: str) -> str:
    return path.replace("gcp://", "").replace("gcs://", "").replace("gs://", "")


class GCloudCredentials:
    def __init__(self, token: Union[str, Dict] = None, project: str = None):
        self.scope = "https://www.googleapis.com/auth/cloud-platform"
        self.project = project
        self.credentials = None
        self.token = token
        self.tokens: Dict[str, Dict] = {}
        self.connect(method=token)

    def _load_tokens(self):
        """Get "browser" tokens from disc"""
        try:
            with open(".gcs_token", "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def _save_tokens(self, token):
        with open(".gcs_token", "wb") as f:
            pickle.dump(token, f, 2)

    def _connect_google_default(self):
        """Attempts to get credentials from google default configurations:
            environment variable GOOGLE_APPLICATION_CREDENTIALS, Google Cloud SDK default credentials or default project.
            For more details see: https://google-auth.readthedocs.io/en/master/reference/google.auth.html#google.auth.default

        Raises:
            ValueError: If the name of the default project doesn't match the GCSProvider project name.
            DefaultCredentialsError: If no credentials are found.
        """
        credentials, project = gauth.default(scopes=[self.scope])
        if self.project and self.project != project:
            raise ValueError(
                "Project name does not match the google default project name."
            )
        self.project = project
        self.credentials = credentials

    def _connect_cache(self):
        """Load token stored after using _connect_browser() method."""
        credentials = self._load_tokens()
        if credentials:
            self.credentials = credentials

    def _dict_to_credentials(self, token: Dict):
        """
        Convert dict-style token.
        Does not preserve access token itself, assumes refresh required.

        Args:
            token (Dict): dictionary with token to be stored in .json file.

        Returns:
            Path to stored .json file.
        """
        token_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        with open(token_file.name, "w") as f:
            json.dump(token, f)
        return token_file.name

    def _connect_token(self, token: Union[str, Dict] = None):
        """
        Connect using a concrete token.

        Args:
            token (str/dict/Credentials):
                If a str, try to load as a Service file, or next as a JSON; if
                dict, try to interpret as credentials; if Credentials, use directly.

        Raises:
            FileNotFoundError: If token file doesn't exist.
            ValueError: If token format isn't supported by gauth.
        """
        if isinstance(token, str):
            if not os.path.exists(token):
                raise FileNotFoundError(token)
            try:
                self._connect_service(token)
                return
            except:
                token = json.load(open(token))
        if isinstance(token, dict):
            token = self._dict_to_credentials(token)
            self._connect_service(token)
            return
        elif isinstance(token, google.auth.credentials.Credentials):
            credentials = token
        else:
            raise ValueError("Token format not understood")
        self.credentials = credentials

    def _connect_service(self, fn):
        credentials = service_account.Credentials.from_service_account_file(
            fn, scopes=[self.scope]
        )
        self.credentials = credentials

    def _connect_browser(self):
        """Create and store new credentials using OAuth authentication method.
            Requires having default client configuration file in ~/.config/gcloud/application_default_credentials.json
            (default location after initializing gcloud).

        Raises:
            GCSDefaultCredsNotFoundError: if application deafault credentials can't be found.
        """
        try:
            if os.name == "nt":
                path = os.path.join(
                    os.getenv("APPDATA"), "gcloud/application_default_credentials.json"
                )
            else:
                path = posixpath.expanduser(
                    "~/.config/gcloud/application_default_credentials.json"
                )
            with open(path, "r") as f:
                default_config = json.load(f)
        except:
            raise GCSDefaultCredsNotFoundError()
        client_config = {
            "installed": {
                "client_id": default_config["client_id"],
                "client_secret": default_config["client_secret"],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://accounts.google.com/o/oauth2/token",
            }
        }
        flow = InstalledAppFlow.from_client_config(client_config, [self.scope])
        credentials = flow.run_console()
        self._save_tokens(credentials)
        self.credentials = credentials

    def _connect_anon(self):
        """Use provider without specific credentials. Applicable for public projects/buckets."""
        self.credentials = None

    def connect(self, method: Union[str, Dict] = None):
        """
        Establish session token. A new token will be requested if the current
        one is within 100s of expiry.

        Args:
            method (str/Dict): Supported methods: ``google_default | cache | anon | browser | None``.
                Type of authorisation to implement - calls `_connect_*` methods.
                If None, will try sequence of methods.

        Raises:
            AttributeError: If method is invalid.
        """
        if method not in [
            "google_default",
            "cache",
            "anon",
            "browser",
            None,
        ]:
            self._connect_token(method)
        elif method is None:
            for meth in ["google_default", "cache", "anon"]:
                self.connect(method=meth)
                break
        elif isinstance(method, str):
            self.__getattribute__("_connect_" + method)()
        else:
            raise AttributeError(f"Invalid method: {method}")


class GCSProvider(StorageProvider):
    """Provider class for using GC storage."""

    def __init__(self, root: str, token: Union[str, Dict] = None, project: str = None):
        """Initializes the GCSProvider.

        Example:
            >>> gcs_provider = GCSProvider("gcs://my-bucket/gcs_ds")

        Args:
            root (str): The root of the provider. All read/write request keys will be appended to root.
            token (str/Dict): GCP token, used for fetching credentials for storage).
                Can be a path to the credentials file, actual credential dictionary or one of the folowing:
                - ``google_default``: Tries to load default credentials for the specified project.
                - ``cache``: Retrieves the previously used credentials from cache if exist.
                - ``anon``: Sets ``credentials=None``.
                - ``browser``: Generates and stores new token file using cli.
            project (str): Name of the project from GCloud.

        Raises:
            ModuleNotFoundError: If google cloud packages aren't installed.
        """
        if not _GOOGLE_PACKAGES_INSTALLED:
            raise ModuleNotFoundError(
                "Google cloud packages are not installed. Run `pip install hub[gcp]`."
            )
        self.root = root
        self.token: Union[str, Dict, None] = token
        self.tag: Optional[str] = None
        self.project = project
        self.missing_exceptions = (
            FileNotFoundError,
            IsADirectoryError,
            NotADirectoryError,
            AttributeError,
            NotFound,
        )
        self._initialize_provider()
        self._presigned_urls: Dict[str, Tuple[str, float]] = {}
        self.expiration: Optional[str] = None

    def subdir(self, path: str):
        return self.__class__(
            root=posixpath.join(self.root, path),
            token=self.token,
            project=self.project,
        )

    def _initialize_provider(self):
        self._set_bucket_and_path()
        if not self.token:
            self.token = None
        self.scoped_credentials = GCloudCredentials(self.token, project=self.project)
        self.retry = retry.Retry(deadline=60)
        self.client = storage.Client(credentials=self.scoped_credentials.credentials)
        self._client_bucket = None

    @property
    def client_bucket(self):
        if self._client_bucket is None:
            self._client_bucket = self.client.get_bucket(self.bucket)
        return self._client_bucket

    def _set_bucket_and_path(self):
        root = _remove_protocol_from_path(self.root)
        split_root = root.split("/", 1)
        self.bucket = split_root[0]
        if len(split_root) > 1:
            self.path = split_root[1]
        else:
            self.path = ""
        if not self.path.endswith("/"):
            self.path += "/"

    def _get_path_from_key(self, key):
        return posixpath.join(self.path, key)

    def _all_keys(self):
        self._blob_objects = self.client_bucket.list_blobs(prefix=self.path)
        return {posixpath.relpath(obj.name, self.path) for obj in self._blob_objects}

    def _set_hub_creds_info(self, hub_path: str, expiration: str):
        """Sets the tag and expiration of the credentials. These are only relevant to datasets using Hub storage.
        This info is used to fetch new credentials when the temporary 12 hour credentials expire.

        Args:
            hub_path (str): The hub cloud path to the dataset.
            expiration (str): The time at which the credentials expire.
        """
        self.hub_path = hub_path
        self.tag = hub_path[6:]  # removing the hub:// part from the path
        self.expiration = expiration

    def clear(self, prefix=""):
        """Remove all keys with given prefix below root - empties out mapping.

        Warning:
            Exercise caution!
        """
        self.check_readonly()
        path = posixpath.join(self.path, prefix) if prefix else self.path
        blob_objects = self.client_bucket.list_blobs(prefix=path)
        for blob in blob_objects:
            try:
                blob.delete()
            except Exception:
                pass

    def rename(self, root):
        """Rename root folder."""
        self.check_readonly()
        path = _remove_protocol_from_path(root)
        new_bucket, new_path = path.split("/", 1)
        if new_bucket != self.client_bucket.name:
            raise RenameError
        blob_objects = self.client_bucket.list_blobs(prefix=self.path)
        dest_objects = self.client_bucket.list_blobs(prefix=new_path)
        for blob in dest_objects:
            raise PathNotEmptyException(use_hub=False)
        for blob in blob_objects:
            new_key = "/".join([new_path, posixpath.relpath(blob.name, self.path)])
            self.client_bucket.rename_blob(blob, new_key)

        self.root = root
        self.path = new_path
        if not self.path.endswith("/"):
            self.path += "/"

    def __getitem__(self, key):
        """Retrieve data."""
        return self.get_bytes(key)

    def get_bytes(
        self,
        path: str,
        start_byte: Optional[int] = None,
        end_byte: Optional[int] = None,
    ):
        """Gets the object present at the path within the given byte range.

        Args:
            path (str): The path relative to the root of the provider.
            start_byte (int, optional): If only specific bytes starting from ``start_byte`` are required.
            end_byte (int, optional): If only specific bytes up to ``end_byte`` are required.

        Returns:
            bytes: The bytes of the object present at the path within the given byte range.

        Raises:
            InvalidBytesRequestedError: If ``start_byte`` > ``end_byte`` or ``start_byte`` < 0 or ``end_byte`` < 0.
            KeyError: If an object is not found at the path.
        """
        try:
            blob = self.client_bucket.get_blob(self._get_path_from_key(path))
            if end_byte is not None:
                end_byte -= 1
            return blob.download_as_bytes(
                retry=self.retry, start=start_byte, end=end_byte
            )
        except self.missing_exceptions:
            raise KeyError(path)

    def __setitem__(self, key, value):
        """Store value in key."""
        self.check_readonly()
        blob = self.client_bucket.blob(self._get_path_from_key(key))
        if isinstance(value, memoryview):
            value = value.tobytes()
        elif isinstance(value, bytearray):
            value = bytes(value)
        # if isinstance(value, memoryview) and (
        #     value.strides == (1,) and value.shape == (len(value.obj),)
        # ):
        #     value = value.obj
        # value = bytes(value)
        blob.upload_from_string(value, retry=self.retry)

    def __iter__(self):
        """Iterating over the structure."""
        yield from [f for f in self._all_keys() if not f.endswith("/")]

    def __len__(self):
        """Returns length of the structure."""
        return len(self._all_keys())

    def __delitem__(self, key):
        """Remove key."""
        self.check_readonly()
        blob = self.client_bucket.blob(self._get_path_from_key(key))
        try:
            blob.delete()
        except self.missing_exceptions:
            raise KeyError(key)

    def __contains__(self, key):
        """Checks if key exists in mapping."""
        stats = storage.Blob(
            bucket=self.client_bucket, name=self._get_path_from_key(key)
        ).exists(self.client_bucket.client)
        return stats

    def __getstate__(self):
        return (
            self.root,
            self.token,
            self.missing_exceptions,
            self.project,
            self.read_only,
        )

    def __setstate__(self, state):
        self.root = state[0]
        self.token = state[1]
        self.missing_exceptions = state[2]
        self.project = state[3]
        self.read_only = state[4]
        self._initialize_provider()

    def get_presigned_url(self, key, full=False):
        if full:
            root = _remove_protocol_from_path(key)
            split_root = root.split("/", 1)
            bucket = split_root[0]
            key = split_root[1] if len(split_root) > 1 else ""

            client_bucket = self.client.get_bucket(bucket)
        else:
            client_bucket = self.client_bucket

        url = None
        cached = self._presigned_urls.get(key)
        if cached:
            url, t_store = cached
            t_now = time.time()
            if t_now - t_store > 3200:
                del self._presigned_urls[key]
                url = None

        if url is None:
            if self._is_hub_path:
                client = HubBackendClient(self.token)  # type: ignore
                org_id, ds_name = self.tag.split("/")  # type: ignore
                url = client.get_presigned_url(org_id, ds_name, key)
            else:
                blob = client_bucket.get_blob(
                    self._get_path_from_key(key) if not full else key
                )
                url = blob.generate_signed_url(datetime.timedelta(seconds=3600))
            self._presigned_urls[key] = (url, time.time())
        return url

    def get_object_size(self, key: str) -> int:
        blob = self.client_bucket.get_blob(self._get_path_from_key(key))
        return blob.size

    def get_object_from_full_url(self, url: str):
        root = _remove_protocol_from_path(url)
        split_root = root.split("/", 1)
        bucket = split_root[0]
        path = split_root[1] if len(split_root) > 1 else ""

        client_bucket = self.client.get_bucket(bucket)

        try:
            blob = client_bucket.get_blob(path)
            return blob.download_as_bytes(retry=self.retry)
        except self.missing_exceptions:
            raise KeyError(path)
