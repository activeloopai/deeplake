import posixpath
import pickle
import json
import os
import tempfile
import time
from typing import Dict, Optional, Tuple, Union
from datetime import timezone, datetime, timedelta

from deeplake.util.path import relpath


from deeplake.core.storage.provider import StorageProvider
from deeplake.util.exceptions import (
    GCSDefaultCredsNotFoundError,
    RenameError,
    PathNotEmptyException,
)
from deeplake.client.client import DeepLakeBackendClient


def _remove_protocol_from_path(path: str) -> str:
    return path.replace("gcp://", "").replace("gcs://", "").replace("gs://", "")


class GCloudCredentials:
    def __init__(
        self, token: Optional[Union[str, Dict]] = None, project: Optional[str] = None
    ):
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
        import google.auth as gauth  # type: ignore

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

    def _connect_from_dict(self, token: Dict):
        """
        Connect using a dictionary type token.

        Distinguishes between service account and user account tokens.
        Args:
            token (Dict): dictionary with token to be stored in either .json Service file, or credentials combination.
        """
        from google.oauth2.credentials import Credentials  # type: ignore

        if "gcs_oauth_token" in token:
            self.credentials = Credentials(token["gcs_oauth_token"])
            self.project = token.get("project_id", None)
            return

        service_file = self._dict_to_credentials(token)
        self._connect_service(service_file)

    def _connect_token(self, token: Optional[Union[str, Dict]] = None):
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
        import google.auth.credentials  # type: ignore

        if isinstance(token, str):
            if not os.path.exists(token):
                raise FileNotFoundError(token)
            try:
                self._connect_service(token)
                return
            except:
                token = json.load(open(token))
        if isinstance(token, dict):
            self._connect_from_dict(token)
            return
        elif isinstance(token, google.auth.credentials.Credentials):
            credentials = token
        else:
            raise ValueError("Token format not understood")

        self.credentials = credentials

    def _connect_service(self, fn):
        from google.oauth2 import service_account  # type: ignore

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
        from google_auth_oauthlib.flow import InstalledAppFlow  # type: ignore

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

    def connect(self, method: Optional[Union[str, Dict]] = None):
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

    def get_token_info(self):
        token = self.token
        if token in ["browser", "cache"]:
            raise NotImplementedError("Token info not available for browser method")
        if token == "anon":
            return {"anon": "anon"}
        if token == "google_default":
            token = os.env.get("GOOGLE_APPLICATION_CREDENTIALS")
            if token is None:
                raise ValueError(
                    "Token info not found for google_default method as env variable GOOGLE_APPLICATION_CREDENTIALS is not set"
                )
        if isinstance(token, str):
            if not os.path.exists(token):
                raise FileNotFoundError(token)
            with open(token, "r") as f:
                token = f.read()
            return {"json_credentials": token}
        if isinstance(token, dict):
            return {"json_credentials": json.dumps(token)}
        return {}

    def are_credentials_downscoped(self):
        return self.token is not None and "gcs_oauth_token" in self.token


class GCSProvider(StorageProvider):
    """Provider class for using GC storage."""

    def __init__(
        self,
        root: str,
        creds: Optional[Union[str, Dict]] = None,
        token: Optional[str] = None,
        project: Optional[str] = None,
    ):
        """Initializes the GCSProvider.

        Example:
            >>> gcs_provider = GCSProvider("gcs://my-bucket/gcs_ds")

        Args:
            root (str): The root of the provider. All read/write request keys will be appended to root.
            creds (str/Dict): GCP creds, used for fetching credentials for storage).
                Can be a path to the credentials file, actual credential dictionary or one of the folowing:
                - ``google_default``: Tries to load default credentials for the specified project.
                - ``cache``: Retrieves the previously used credentials from cache if exist.
                - ``anon``: Sets ``credentials=None``.
                - ``browser``: Generates and stores new creds file using cli.
            token (str, optional): Activeloop token, used for fetching credentials for Deep Lake datasets (if this is underlying storage for Deep Lake dataset).
                This is optional, tokens are normally autogenerated.
            project (str): Name of the project from GCloud.

        Raises:
            ModuleNotFoundError: If google cloud packages aren't installed.
        """

        try:
            import google.cloud.storage  # type: ignore
            import google.api_core  # type: ignore
            import google.oauth2  # type: ignore
            import google.auth  # type: ignore
            import google_auth_oauthlib  # type: ignore
            from google.api_core.exceptions import NotFound  # type: ignore
        except ImportError:
            raise ModuleNotFoundError(
                "Google cloud packages are not installed. Run `pip install deeplake[gcp]`."
            )

        self.root = root
        self.token: Union[str, Dict, None] = creds
        self.activeloop_token = token
        self.tag: Optional[str] = None
        self.project = project
        self.missing_exceptions = (
            FileNotFoundError,
            IsADirectoryError,
            NotADirectoryError,
            AttributeError,
            NotFound,
        )
        self.org_id: Optional[str] = None
        self.creds_key: Optional[str] = None
        self._initialize_provider()
        self._presigned_urls: Dict[str, Tuple[str, float]] = {}
        self.expiration: Optional[str] = None
        self.db_engine: bool = False
        self.repository: Optional[str] = None

    def subdir(self, path: str, read_only: bool = False):
        sd = self.__class__(
            root=posixpath.join(self.root, path),
            creds=self.token,
            token=self.activeloop_token,
            project=self.project,
        )
        if hasattr(self, "expiration") and self.expiration:
            sd._set_hub_creds_info(
                self.hub_path, self.expiration, self.db_engine, self.repository
            )
        sd.read_only = read_only
        return sd

    def _check_update_creds(self, force=False):
        """If the client has an expiration time, check if creds are expired and fetch new ones.
        This would only happen for datasets stored on Deep Lake storage for which temporary credentials are generated.
        """

        if (
            hasattr(self, "expiration")
            and self.expiration
            and (
                force or float(self.expiration) < datetime.now(timezone.utc).timestamp()
            )
        ):
            client = DeepLakeBackendClient(self.activeloop_token)
            org_id, ds_name = self.tag.split("/")

            mode = "r" if self.read_only else "a"

            url, creds, mode, expiration, repo = client.get_dataset_credentials(
                org_id,
                ds_name,
                mode,
                {"enabled": self.db_engine},
                True,
            )
            self.expiration = expiration
            self.repository = repo
            self.token = creds
            self._initialize_provider()

    def _initialize_provider(self):
        from google.cloud import storage  # type: ignore
        from google.api_core import retry  # type: ignore

        # In case the storage provider is used with Managed Credentials
        if isinstance(self.token, dict):
            self.org_id = self.token.pop("org_id", None)
            self.creds_key = self.token.pop("creds_key", None)

        self._set_bucket_and_path()
        if not self.token:
            self.token = None
        self.scoped_credentials = GCloudCredentials(self.token, project=self.project)
        self.retry = retry.Retry(deadline=60)
        kwargs = {}
        if self.scoped_credentials.project:
            kwargs["project"] = self.scoped_credentials.project

        self.client = storage.Client(
            credentials=self.scoped_credentials.credentials, **kwargs
        )
        self._client_bucket = None

    @property
    def creds_key(self) -> Optional[str]:
        return self._creds_key

    @creds_key.setter
    def creds_key(self, creds_key: Optional[str]):
        self._creds_key = creds_key

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
        self._check_update_creds()
        self._blob_objects = self.client_bucket.list_blobs(prefix=self.path)
        return {posixpath.relpath(obj.name, self.path) for obj in self._blob_objects}

    def _set_hub_creds_info(
        self,
        hub_path: str,
        expiration: str,
        db_engine: bool = True,
        repository: Optional[str] = None,
    ):
        """Sets the tag and expiration of the credentials. These are only relevant to datasets using Deep Lake storage.
        This info is used to fetch new credentials when the temporary 12 hour credentials expire.

        Args:
            hub_path (str): The deeplake cloud path to the dataset.
            expiration (str): The time at which the credentials expire.
            db_engine (bool): Whether Activeloop DB Engine enabled.
            repository (str, Optional): Backend repository where the dataset is stored.
        """
        self.hub_path = hub_path
        self.tag = hub_path[6:]  # removing the hub:// part from the path
        self.expiration = expiration
        self.db_engine = db_engine
        self.repository = repository

    def clear(self, prefix=""):
        """Remove all keys with given prefix below root - empties out mapping.

        Warning:
            Exercise caution!
        """
        self.check_readonly()
        self._check_update_creds()
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
        self._check_update_creds()
        path = _remove_protocol_from_path(root)
        new_bucket, new_path = path.split("/", 1)
        if new_bucket != self.client_bucket.name:
            raise RenameError
        blob_objects = self.client_bucket.list_blobs(prefix=self.path)
        dest_objects = self.client_bucket.list_blobs(prefix=new_path)
        for blob in dest_objects:
            raise PathNotEmptyException(use_hub=False)
        for blob in blob_objects:
            new_key = "/".join([new_path, relpath(blob.name, self.path)])
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
        self._check_update_creds()
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
        self._check_update_creds()
        blob = self.client_bucket.blob(self._get_path_from_key(key))
        if isinstance(value, memoryview):
            value = value.tobytes()
        elif isinstance(value, bytearray):
            value = bytes(value)
        blob.upload_from_string(value, retry=self.retry)

    def __iter__(self):
        """Iterating over the structure."""
        yield from [f for f in self._all_keys() if not f.endswith("/")]

    def __len__(self):
        """Returns length of the structure."""
        self._check_update_creds()
        return len(self._all_keys())

    def __delitem__(self, key):
        """Remove key."""
        self.check_readonly()
        self._check_update_creds()
        blob = self.client_bucket.blob(self._get_path_from_key(key))
        try:
            blob.delete()
        except self.missing_exceptions:
            raise KeyError(key)

    def __contains__(self, key):
        """Checks if key exists in mapping."""
        from google.cloud import storage  # type: ignore

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
            self.db_engine,
            self.repository,
        )

    def __setstate__(self, state):
        self.root = state[0]
        self.token = state[1]
        self.missing_exceptions = state[2]
        self.project = state[3]
        self.read_only = state[4]
        self.db_engine = state[5]
        self.repository = state[6]
        self._initialize_provider()

    def get_presigned_url(self, key, full=False):
        """
        Generate a presigned URL for accessing an object in GCS.

        Args:
            key (str): The key for the object.
            full (bool): Whether the key is a full path or relative to the root.

        Returns:
            str: The presigned URL.
        """
        self._check_update_creds()
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
                client = DeepLakeBackendClient(self.activeloop_token)  # type: ignore
                org_id, ds_name = self.tag.split("/")  # type: ignore
                url = client.get_presigned_url(org_id, ds_name, key)
            else:
                if self.scoped_credentials.are_credentials_downscoped():
                    client = DeepLakeBackendClient(self.activeloop_token)
                    url = client.get_blob_presigned_url(
                        org_id=self.org_id,
                        creds_key=self.creds_key,
                        blob_path=f"gcs://{root}",
                    )
                else:
                    blob = client_bucket.blob(
                        self._get_path_from_key(key) if not full else key
                    )
                    url = blob.generate_signed_url(expiration=timedelta(hours=1))

            self._presigned_urls[key] = (url, time.time())

        return url

    def get_object_size(self, key: str) -> int:
        self._check_update_creds()
        blob = self.client_bucket.get_blob(self._get_path_from_key(key))
        return blob.size

    def get_object_from_full_url(self, url: str):
        self._check_update_creds()
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

    def get_creds(self):
        if self.scoped_credentials.are_credentials_downscoped:
            d = self.scoped_credentials.token
        else:
            d = self.scoped_credentials.get_token_info()

        d["expiration"] = (
            self.expiration if hasattr(self, "expiration") and self.expiration else ""
        )
        return d
