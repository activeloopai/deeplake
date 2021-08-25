import posixpath
import json
import os
from typing import Dict, Union
import textwrap

from google.cloud import storage  # type: ignore
from google.oauth2 import service_account  # type: ignore
import google.auth as gauth  # type: ignore
import google.auth.compute_engine  # type: ignore
import google.auth.credentials  # type: ignore
import google.auth.exceptions  # type: ignore
from google.oauth2.credentials import Credentials  # type: ignore

from hub.core.storage.provider import StorageProvider


class GoogleCredentials:
    def __init__(self, token, project=None):
        self.scope = "https://www.googleapis.com/auth/cloud-platform"
        self.project = project
        self.access = "full-access"
        self.heads = {}
        self.credentials = None
        self.method = None
        self.token = token
        self.connect(method=token)

    def _connect_google_default(self):
        credentials, project = gauth.default(scopes=[self.scope])
        msg = textwrap.dedent(
            """\
        User-provided project '{}' does not match the google default project '{}'. Either
          1. Accept the google-default project by not passing a `project` to GCSFileSystem
          2. Configure the default project to match the user-provided project (gcloud config set project)
          3. Use an authorization method other than 'google_default' by providing 'token=...'
        """
        )
        if self.project and self.project != project:
            raise ValueError(msg.format(self.project, project))
        self.project = project
        self.credentials = credentials

    def _connect_cloud(self):
        self.credentials = gauth.compute_engine.Credentials()

    def _connect_cache(self):
        project, access = self.project, self.access
        if (project, access) in self.tokens:
            credentials = self.tokens[(project, access)]
            self.credentials = credentials

    def _dict_to_credentials(self, token):
        """
        Convert old dict-style token.
        Does not preserve access token itself, assumes refresh required.
        """
        token_path = posixpath.expanduser("gcs.json")
        with open(token_path, "w") as f:
            json.dump(token, f)
        return token_path

    def _connect_token(self, token):
        """
        Connect using a concrete token
        Parameters
        ----------
        token: str, dict or Credentials
            If a str, try to load as a Service file, or next as a JSON; if
            dict, try to interpret as credentials; if Credentials, use directly.
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
        if self.credentials.valid:
            self.credentials.apply(self.heads)

    def _connect_service(self, fn):
        credentials = service_account.Credentials.from_service_account_file(
            fn, scopes=[self.scope]
        )
        self.credentials = credentials

    def _connect_anon(self):
        self.credentials = None

    def connect(self, method=None):
        """
        Establish session token. A new token will be requested if the current
        one is within 100s of expiry.
        Parameters
        ----------
        method: str (google_default|cache|cloud|token|anon|browser) or None
            Type of authorisation to implement - calls `_connect_*` methods.
            If None, will try sequence of methods.
        """
        if method not in [
            "google_default",
            "cache",
            "cloud",
            "token",
            "anon",
            None,
        ]:
            self._connect_token(method)
        elif method is None:
            for meth in ["google_default", "cache", "cloud", "anon"]:
                self.connect(method=meth)
                break
        else:
            self.__getattribute__("_connect_" + method)()
            self.method = method


class GCSProvider(StorageProvider):
    """Provider class for using GC storage."""

    def __init__(
        self,
        root: str,
        token: Union[str, Dict] = None,
    ):
        """Initializes the GCSProvider

        Example:
            gcs_provider = GCSProvider("snark-test/gcs_ds")

        Args:
            root (str): The root of the provider. All read/write request keys will be appended to root.
            token (str/Dict): GCP token, used for fetching credentials for storage).
        """
        self.root = root
        self.token: Union[str, Dict, None] = token
        self.missing_exceptions = (
            FileNotFoundError,
            IsADirectoryError,
            NotADirectoryError,
            AttributeError,
        )
        self._initialize_provider()

    def _initialize_provider(self):
        self._set_bucket_and_path()
        if not self.token:
            self.token = None
        scoped_credentials = GoogleCredentials(self.token)
        client = storage.Client(credentials=scoped_credentials.credentials)
        self.client_bucket = client.get_bucket(self.bucket)

    def _set_bucket_and_path(self):
        root = self.root.replace("gcp://", "").replace("gcs://", "")
        self.bucket = root.split("/")[0]
        self.path = root
        if not self.path.endswith("/"):
            self.path += "/"

    def _get_path_from_key(self, key):
        return posixpath.join(self.path, key)

    def _list_keys(self):
        self._blob_objects = self.client_bucket.list_blobs(prefix=self.path)
        return [obj.name for obj in self._blob_objects]

    def clear(self):
        """Remove all keys below root - empties out mapping"""
        self.check_readonly()
        blob_objects = self.client_bucket.list_blobs(prefix=self.path)
        for blob in blob_objects:
            blob.delete()

    def __getitem__(self, key):
        """Retrieve data"""
        try:
            blob = self.client_bucket.get_blob(self._get_path_from_key(key))
            return blob.download_as_bytes()
        except self.missing_exceptions:
            raise KeyError(key)

    def __setitem__(self, key, value):
        """Store value in key"""
        self.check_readonly()
        blob = self.client_bucket.blob(self._get_path_from_key(key))
        if isinstance(value, memoryview):
            value = value.tobytes()
        elif isinstance(value, bytearray):
            value = bytes(value)
        blob.upload_from_string(value)

    def __iter__(self):
        """Iterating over the structure"""
        yield from [f for f in self._list_keys() if not f.endswith("/")]

    def __len__(self):
        """Returns length of the structure"""
        return len(self._list_keys())

    def __delitem__(self, key):
        """Remove key"""
        self.check_readonly()
        blob = self.client_bucket.blob(self._get_path_from_key(key))
        blob.delete()

    def __contains__(self, key):
        """Does key exist in mapping?"""
        stats = storage.Blob(
            bucket=self.client_bucket, name=self._get_path_from_key(key)
        ).exists(self.client_bucket.client)
        return stats

    def __getstate__(self):
        return (self.root, self.token, self.missing_exceptions)

    def __setstate__(self, state):
        self.root = state[0]
        self.token = state[1]
        self.missing_exceptions = state[2]
        self._initialize_provider()
