from typing import Dict, Any

from hub.defaults import MAX_POOL_CONNECTIONS, MAX_CONNECTION_WORKERS
from s3fs import S3FileSystem
import boto3
import botocore

from hub.store.s3_storage import S3Storage
from concurrent.futures import ThreadPoolExecutor


class S3FileSystemReplacement(S3FileSystem):
    def __init__(
        self,
        key: str = None,
        secret: str = None,
        token: str = None,
        client_kwargs: Dict[str, Any] = None,
    ):
        super().__init__(
            key=key,
            secret=secret,
            token=token,
            client_kwargs=client_kwargs,
            use_listings_cache=False,
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
        self.tpool = ThreadPoolExecutor(MAX_CONNECTION_WORKERS)
        self._closed = False

    def get_mapper(self, root: str, check=False, create=False):
        return S3Storage(
            self,
            self.client,
            self.tpool,
            "s3://" + root,
        )

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def close(self):
        if self._closed:
            raise Exception("This S3FileSystemReplacement instance is closed!")
        self.tpool.shutdown(wait=False)
        self._closed = True
