from typing import Optional, Union, Tuple

from .hub_bucket import HubBucket
from .hub_backend import HubBackend
from hub import HubBackendImpl
from hub.storage import recursive as RecursiveStorage
from hub.storage import Storage

def amazon_s3(bucket: str, aws_creds_filepath: Optional[str], aws_key_id: Optional[str], aws_secret_key: Optional[str]) -> HubBackend:
    raise NotImplementedError()

# def GoogleGS(bucket: str, ...)
# Hello World

def filesystem(dir: str) -> HubBackend:
    raise NotImplementedError()

def bucket(backends: Union[HubBackend, Tuple[HubBackend, ...]]) -> HubBucket:
    if backends is HubBackend:
        backends = [backends]

    backends: Tuple[HubBackendImpl, ...] = backends
    
    storage: Storage = None

    for backend in backends:
        if storage is None:
            storage = backend.storage
        else:
            storage = RecursiveStorage(backend.storage, storage)



# bucket = hub.AmazonS3() 
# bucket.array_create()