from typing import Optional, Union, Tuple
import os

from .hub_bucket import HubBucket
from .hub_backend import HubBackend
from hub.hub_bucket_impl import HubBucketImpl
from hub.hub_backend_impl import HubBackendImpl
from hub.storage import recursive as RecursiveStorage
from hub.storage.amazon_s3 import AmazonS3
from hub.storage.filesystem_storage import FileSystemStorage
from hub.storage import Storage
from hub.utils.store_control import StoreControlClient
from hub.exceptions import FileSystemException, S3Exception
from hub.config import CACHE_FILE_PATH


def amazon_s3(
    bucket: Optional[str] = None,
    aws_creds_filepath: Optional[str] = None,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
) -> HubBackend:
    if aws_access_key_id is not None or aws_secret_access_key is not None:
        assert aws_access_key_id is not None
        assert aws_secret_access_key is not None
        assert bucket is not None
        assert aws_creds_filepath is None
        return HubBackendImpl(
            AmazonS3(bucket, aws_access_key_id, aws_secret_access_key)
        )
    elif aws_creds_filepath is not None:
        raise NotImplementedError()
    else:
        config = StoreControlClient.get_config()
        if bucket is None:
            bucket = config["BUCKET"]

        aws_access_key_id = config["AWS_ACCESS_KEY_ID"]
        aws_secret_access_key = config["AWS_SECRET_ACCESS_KEY"]

        return HubBackendImpl(
            AmazonS3(bucket, aws_access_key_id, aws_secret_access_key)
        )


# def GoogleGS(bucket: str, ...)
# Hello World


def filesystem(dir: str) -> HubBackend:
    dir = os.path.expanduser(dir)
    assert os.path.isdir(dir)

    return HubBackendImpl(FileSystemStorage(dir))


def bucket(backends: Union[HubBackend, Tuple[HubBackend, ...]]) -> HubBucket:
    if isinstance(backends, HubBackend):
        backends = [backends]

    backends: Tuple[HubBackendImpl, ...] = backends

    storage: Storage = None

    for backend in backends:
        if storage is None:
            storage = backend.storage
        else:
            storage = RecursiveStorage(backend.storage, storage)

    return HubBucketImpl(storage)


# bucket = hub.AmazonS3()
# bucket.array_create()
