import posixpath
import json
import os
from typing import Dict, Union

from hub.core.storage.provider import StorageProvider
from google.cloud import storage  # type: ignore
from google.oauth2 import service_account  # type: ignore


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
        if self.token:
            if isinstance(self.token, dict):
                token_path = posixpath.expanduser("gcs.json")
                with open(token_path, "wb") as f:
                    json.dump(self.token, f)
                self.token = token_path
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.token
            credentials = service_account.Credentials.from_service_account_file(
                self.token
            )

            scoped_credentials = credentials.with_scopes(
                ["https://www.googleapis.com/auth/cloud-platform"]
            )

            client = storage.Client(credentials=scoped_credentials)
        else:
            client = storage.Client(credentials=self.token)
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
