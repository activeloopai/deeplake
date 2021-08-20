import posixpath
from typing import Dict, Optional, Union
from hub.core.storage.provider import StorageProvider
import gcsfs  # type: ignore
import json


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
        self.token: Union[str, Dict] = token
        self.missing_exceptions = (
            FileNotFoundError,
            IsADirectoryError,
            NotADirectoryError,
        )
        self.initialize_provider()

    def initialize_provider(self):
        self._set_bucket_and_path()
        if isinstance(self.token, str):
            with open(self.token) as token_file:
                self.token = json.load(token_file)
        self.fs = gcsfs.GCSFileSystem(token=self.token)

    def _set_bucket_and_path(self):
        root = self.root.replace("gcp://", "").replace("gcs://", "")
        self.bucket = root.split("/")[0]
        self.path = root
        if not self.path.endswith("/"):
            self.path += "/"

    def clear(self):
        """Remove all keys below root - empties out mapping"""
        self.fs.delete(self.path, True)

    def __getitem__(self, key):
        """Retrieve data"""
        try:
            with self.fs.open(posixpath.join(self.path, key), "rb") as f:
                return f.read()
        except self.missing_exceptions:
            raise KeyError(key)

    def __setitem__(self, key, value):
        """Store value in key"""
        with self.fs.open(posixpath.join(self.path, key), "wb") as f:
            f.write(value)

    def __iter__(self):
        """iterating over the structure"""
        yield from (x for x in self.fs.find(self.root))

    def __len__(self):
        """returns length of the structure"""
        return len(self.fs.find(self.root))

    def __delitem__(self, key):
        """Remove key"""
        self.fs.rm(posixpath.join(self.path, key))

    def __contains__(self, key):
        """Does key exist in mapping?"""
        path = posixpath.join(self.path, key)
        return self.fs.exists(path)
