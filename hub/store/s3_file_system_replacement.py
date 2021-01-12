from typing import Dict, Any

from botocore import endpoint
from hub.defaults import MAX_POOL_CONNECTIONS
from s3fs import S3FileSystem
import boto3
import botocore

from hub.store.s3_storage import S3Storage
from zarr import MemoryStore


class S3FileSystemReplacement(S3FileSystem):
    def __init__(
        self,
        key: str = None,
        secret: str = None,
        token: str = None,
        client_kwargs: Dict[str, Any] = None,
    ):
        super().__init__(
            key=key, secret=secret, token=token, client_kwargs=client_kwargs
        )
        endpoint_url = client_kwargs and client_kwargs.get("endpoint_url") or None
        self.client = boto3.client(
            "s3",
            aws_access_key_id=key,
            aws_secret_access_key=secret,
            aws_session_token=token,
            config=botocore.config.Config(max_pool_connections=MAX_POOL_CONNECTIONS),
            endpoint_url=endpoint_url,
        )
        self.client_kwargs = client_kwargs

    def get_mapper(self, root: str, check=False, create=False):
        return S3Storage(
            self,
            self.client,
            "s3://" + root,
        )
