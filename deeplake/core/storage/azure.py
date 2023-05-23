import posixpath
import time
from typing import Dict, Optional, Set, Tuple
from datetime import datetime, timedelta
from deeplake.core.storage.provider import StorageProvider
from deeplake.client.client import DeepLakeBackendClient

try:
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import (
        BlobServiceClient,
        BlobSasPermissions,
        BlobClient,
        generate_blob_sas,
    )

    _AZURE_PACKAGES_INSTALLED = True
except ImportError:
    _AZURE_PACKAGES_INSTALLED = False


class AzureProvider(StorageProvider):
    def __init__(self, root: str, creds: Optional[Dict] = None):
        if not _AZURE_PACKAGES_INSTALLED:
            raise ImportError(
                "Azure packages not installed. Run `pip install deeplake[azure]`."
            )
        self.root = root
        self._set_clients()
        self._presigned_urls: Dict[str, Tuple[str, float]] = {}
        self.expiration: Optional[str] = None
        self.db_engine: bool = False
        self.repository: Optional[str] = None
        self.creds = creds

    def _get_attrs(self, path: str) -> Tuple[str]:
        split_path = path.replace("az://", "").strip("/").split("/", 2)
        if len(split_path) == 1:
            raise ValueError(
                "Container name must be provided. Path must be in the format az://<account_name>/<container_name>/<root_folder>"
            )
        elif len(split_path) == 2:
            account_name, container_name, root_folder = *split_path, ""
        else:
            account_name, container_name, root_folder = split_path
        return account_name, container_name, root_folder

    def _set_clients(self):
        self.account_name, self.container_name, self.root_folder = self._get_attrs(
            self.root
        )
        self.account_url = f"https://{self.account_name}.blob.core.windows.net"
        self.default_credential = DefaultAzureCredential()
        self.blob_service_client = BlobServiceClient(
            self.account_url, credential=self.default_credential
        )
        self.container_client = self.blob_service_client.get_container_client(
            self.container_name
        )

    def __setitem__(self, path, content):
        self.check_readonly()
        if isinstance(content, memoryview):
            content = content.tobytes()
        elif isinstance(content, bytearray):
            content = bytes(content)
        blob_client = self.container_client.get_blob_client(
            f"{self.root_folder}/{path}"
        )
        blob_client.upload_blob(content, overwrite=True)

    def __getitem__(self, path):
        return self.get_bytes(path)

    def __delitem__(self, path):
        self.check_readonly()
        blob_client = self.container_client.get_blob_client(
            f"{self.root_folder}/{path}"
        )
        if not blob_client.exists():
            raise KeyError(path)
        blob_client.delete_blob()

    def get_bytes(
        self,
        path: str,
        start_byte: Optional[int] = None,
        end_byte: Optional[int] = None,
    ):
        if start_byte is not None and end_byte is not None:
            if start_byte == end_byte:
                return b""
            offset = start_byte
            length = end_byte - start_byte
        elif start_byte is not None:
            offset = start_byte
            length = None
        elif end_byte is not None:
            offset = 0
            length = end_byte
        else:
            offset = 0
            length = None

        blob_client = self.container_client.get_blob_client(
            f"{self.root_folder}/{path}"
        )
        if not blob_client.exists():
            raise KeyError(path)
        byts = blob_client.download_blob(offset=offset, length=length).readall()
        return byts

    def clear(self, prefix=""):
        self.check_readonly()
        blobs = [
            posixpath.join(self.root_folder, key) for key in self._all_keys(prefix)
        ]
        self.container_client.delete_blobs(*blobs)

    def _all_keys(self, prefix: str = ""):
        prefix = posixpath.join(self.root_folder, prefix)
        return {
            posixpath.relpath(blob.name, self.root_folder)
            for blob in self.container_client.list_blobs(
                name_starts_with=prefix, include=["metadata"]
            )
            if not (blob.get("metadata") or {}).get(
                "hdi_isfolder", False
            )  # https://github.com/Azure/azure-sdk-for-python/issues/24814
        }

    def __iter__(self):
        yield from self._all_keys()

    def __len__(self):
        return len(self._all_keys())

    def __getstate__(self):
        return {
            "root": self.root,
            "read_only": self.read_only,
            "db_engine": self.db_engine,
            "repository": self.repository,
            "expiration": self.expiration,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._set_clients()

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

    def subdir(self, path: str, read_only: bool = False):
        sd = self.__class__(root=posixpath.join(self.root, path))
        if self.expiration:
            sd._set_hub_creds_info(
                self.hub_path, self.expiration, self.db_engine, self.repository
            )
        sd.read_only = read_only
        return sd

    def rename(self, root: str):
        self.check_readonly()
        account_name, container_name, root_folder = (
            root.replace("az://", "").strip("/").split("/", 2)
        )
        assert (
            account_name == self.account_name
        ), "Cannot rename across storage accounts"
        assert container_name == self.container_name, "Cannot rename across containers"
        for blob_name in self._all_keys():
            source_blob = self.container_client.get_blob_client(blob_name)
            destination_blob = self.container_client.get_blob_client(
                f"{root_folder}/{blob_name}"
            )
            if destination_blob.exists():
                raise ValueError(
                    f"Cannot rename {source_blob.url} to {destination_blob.url} because {destination_blob.url} already exists"
                )
            destination_blob.upload_blob_from_url(
                source_url=source_blob.url, overwrite=False
            )
            source_blob.delete_blob()
        self.root_folder = root_folder

    def get_object_size(self, path: str) -> int:
        blob_client = self.container_client.get_blob_client(
            f"{self.root_folder}/{path}"
        )
        if not blob_client.exists():
            raise KeyError(path)
        return blob_client.get_blob_properties().size

    def get_clients_from_full_path(self, url: str) -> BlobClient:
        account_name, container_name, blob_path = self._get_attrs(url)
        account_url = f"https://{account_name}.blob.core.windows.net"
        blob_service_client = BlobServiceClient(
            account_url, credential=self.default_credential
        )
        blob_client = blob_service_client.get_blob_client(container_name, blob_path)
        if not blob_client.exists():
            raise KeyError(url)
        return blob_client, blob_service_client

    def get_presigned_url(self, path: str, full: bool = False) -> str:
        if full:
            blob_client, blob_service_client = self.get_clients_from_full_path(path)
            account_name = blob_client.account_name
            container_name = blob_client.container_name
            blob_path = blob_client.blob_name
            account_url = f"https://{account_name}.blob.core.windows.net"
        else:
            blob_service_client = self.blob_service_client
            account_name = self.account_name
            container_name = self.container_name
            blob_path = f"{self.root_folder}/{path}"
            account_url = self.account_url

        url = None
        cached = self._presigned_urls.get(path)
        if cached:
            url, t_store = cached
            t_now = time.time()
            if t_now - t_store > 3200:
                del self._presigned_urls[path]
                url = None

        if url is None:
            if self._is_hub_path:
                assert not full
                client = DeepLakeBackendClient(self.token)  # type: ignore
                org_id, ds_name = self.tag.split("/")  # type: ignore
                url = client.get_presigned_url(org_id, ds_name, path)
            else:
                user_delegation_key = blob_service_client.get_user_delegation_key(
                    datetime.utcnow(), datetime.utcnow() + timedelta(hours=1)
                )
                sas_token = generate_blob_sas(
                    account_name,
                    container_name,
                    blob_path,
                    user_delegation_key=user_delegation_key,
                    permission=BlobSasPermissions(read=True),
                    expiry=datetime.utcnow() + timedelta(hours=1),
                )
                url = f"{account_url}/{container_name}/{blob_path}?{sas_token}"
            self._presigned_urls[path] = (url, time.time())

        return url

    def get_object_from_full_url(self, url: str) -> bytes:
        blob_client, _ = self.get_clients_from_full_path(url)
        return blob_client.download_blob().readall()
