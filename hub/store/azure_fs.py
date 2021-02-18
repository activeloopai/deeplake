"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from fsspec import AbstractFileSystem
import array
from collections.abc import MutableMapping
from azure.storage.blob import BlobServiceClient


class AzureBlobFileSystem(AbstractFileSystem):
    def __init__(
        self,
        account_name: str,
        account_key: str = None,
        connection_string: str = None,
        credential: str = None,
        sas_token: str = None,
        request_session=None,
        socket_timeout: int = None,
        client_id: str = None,
        client_secret: str = None,
        tenant_id: str = None,
    ):

        super().__init__()
        self.account_name = account_name
        self.account_key = account_key
        self.connection_string = connection_string
        self.credential = credential
        self.sas_token = sas_token
        self.request_session = request_session
        self.socket_timeout = socket_timeout
        self.client_id = client_id
        self.client_secret = client_secret
        self.tenant_id = tenant_id
        if (
            self.credential is None
            and self.account_key is None
            and self.sas_token is None
            and self.client_id is not None
        ):
            self.credential = self._get_credential_from_service_principal()
        self.do_connect()

    def _get_credential_from_service_principal(self):
        """
        Create a Credential for authentication.  This can include a TokenCredential
        client_id, client_secret and tenant_id

        Returns
        -------
        Credential
        """
        from azure.identity import ClientSecretCredential

        sp_token = ClientSecretCredential(
            tenant_id=self.tenant_id,
            client_id=self.client_id,
            client_secret=self.client_secret,
        )
        return sp_token

    def do_connect(self):
        """Connect to the BlobServiceClient, using user-specified connection details.
        Tries credentials first, then connection string and finally account key

        Raises
        ------
        ValueError if none of the connection details are available
        """
        try:
            self.account_url: str = f"https://{self.account_name}.blob.core.windows.net"
            if self.credential is not None:
                self.service_client = BlobServiceClient(
                    account_url=self.account_url, credential=self.credential
                )
            elif self.connection_string is not None:
                self.service_client = BlobServiceClient.from_connection_string(
                    conn_str=self.connection_string
                )
            elif self.account_key is not None:
                self.service_client = BlobServiceClient(
                    account_url=self.account_url, credential=self.account_key
                )
            elif self.sas_token is not None:
                self.service_client = BlobServiceClient(
                    account_url=self.account_url + self.sas_token, credential=None
                )
            else:
                self.service_client = BlobServiceClient(account_url=self.account_url)

        except Exception as e:
            raise ValueError(f"unable to connect to account for {e}")

    def exists(self, path):
        """
        Checks whether the given path exists in the File System

        Returns
        -------
        Boolean
        """
        split_path = path.split("/")
        container_name = split_path[0]
        sub_path = "/".join(split_path[1:])
        container = self.service_client.get_container_client(container_name)
        it = container.list_blobs(name_starts_with=sub_path)
        return len(list(it)) > 0

    def ls(self, path, refresh=True):
        """
        Finds all the files in the given path in the File System
        Returns
        -------
        List of full paths of all files found in given path
        """
        return self.find(path)

    def isfile(self, path):
        """Is this entry file-like?
        Azure fs only stores path to files and not folders. This is always true
        """
        return True

    def find(self, path):
        """
        Finds all the files in the given path in the File System

        Returns
        -------
        List of full paths of all files found in given path
        """
        split_path = path.split("/")
        container_name = split_path[0]
        sub_path = "/".join(split_path[1:])
        container = self.service_client.get_container_client(container_name)
        it = container.list_blobs(name_starts_with=sub_path)
        return [f"{container_name}/{item['name']}" for item in it]

    def rm(self, path, recursive=False, maxdepth=None):
        """Removes all the files in the given path"""
        split_path = path.split("/")
        container_name = split_path[0]
        sub_path = "/".join(split_path[1:])
        container = self.service_client.get_container_client(container_name)
        it = container.list_blobs(name_starts_with=sub_path)
        for item in it:
            container.delete_blob(item)

    def makedirs(self, path, exist_ok=False):
        """Recursively creates directories in path"""
        # in azure empty directories have no meaning, so makedirs not needed
        return

    def get_mapper(self, root, check=False, create=False):
        """Create key-value interface for given root"""
        return FSMap(root, self)

    def upload(self, path, value):
        """Uploads value to the given path"""
        split_path = path.split("/")
        container_name = split_path[0]
        sub_path = "/".join(split_path[1:])
        blob_client = self.service_client.get_blob_client(container_name, sub_path)
        blob_client.upload_blob(value, overwrite=True)

    def download(self, path):
        """Downloads the value from the given path"""
        if not self.exists(path):
            raise KeyError()
        split_path = path.split("/")
        container_name = split_path[0]
        sub_path = "/".join(split_path[1:])
        blob_client = self.service_client.get_blob_client(container_name, sub_path)
        return blob_client.download_blob().readall()

    def cat_file(self, path):
        return self.download(path)

    def pipe_file(self, path, value):
        return self.upload(path, value)


class FSMap(MutableMapping):
    def __init__(self, root, fs, check=False, create=False, missing_exceptions=None):
        self.fs = fs
        self.root = root
        if missing_exceptions is None:
            missing_exceptions = (
                FileNotFoundError,
                IsADirectoryError,
                NotADirectoryError,
            )
        self.missing_exceptions = missing_exceptions
        if check and not create:
            if not self.fs.exists(root):
                raise ValueError(
                    "Path %s does not exist. Create "
                    " with the ``create=True`` keyword" % root
                )

    def clear(self):
        """Remove all keys below root - empties out mapping"""
        self.fs.rm(self.root, True)

    def _key_to_str(self, key):
        """Generate full path for the key"""
        if isinstance(key, (tuple, list)):
            key = str(tuple(key))
        else:
            key = str(key)
        return "/".join([self.root, key]) if self.root else key

    def _str_to_key(self, s):
        """Strip path of to leave key name"""
        return s[len(self.root) :].lstrip("/")

    def __getitem__(self, key, default=None):
        """Retrieve data"""
        k = self._key_to_str(key)
        try:
            result = self.fs.download(k)
        except self.missing_exceptions:
            if default is not None:
                return default
            raise KeyError(key)
        return result

    def __setitem__(self, key, value):
        """Store value in key"""
        key = self._key_to_str(key)
        if isinstance(value, array.array) or isinstance(value, memoryview):
            value = bytearray(value)
        self.fs.upload(key, value)

    def __iter__(self):
        """iterating over the structure"""
        return (self._str_to_key(x) for x in self.fs.find(self.root))

    def __len__(self):
        """returns length of the structure"""
        return len(self.fs.find(self.root))

    def __delitem__(self, key):
        """Remove key"""
        self.fs.rm(self._key_to_str(key))

    def __contains__(self, key):
        """Does key exist in mapping?"""
        path = self._key_to_str(key)
        return self.fs.exists(path)
