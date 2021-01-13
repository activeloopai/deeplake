from concurrent.futures.thread import ThreadPoolExecutor
from hub.defaults import MAX_CONNECTION_WORKERS
import pytest
from s3fs import S3FileSystem
import cloudpickle

import boto3

from hub.store.s3_storage import S3Storage
from hub.utils import s3_creds_exist


def create_client():
    return boto3.client("s3")


@pytest.mark.skipif(not s3_creds_exist(), reason="Requires s3 credentials")
def test_s3_storage():
    _storage = S3FileSystem()
    with ThreadPoolExecutor(1) as pool:
        storage = S3Storage(
            _storage, create_client(), pool, "s3://snark-test/test_s3_storage"
        )

        storage["hello"] = bytes("world2", "utf-8")
        assert storage["hello"] == bytes("world2", "utf-8")
        assert len(storage) == 1
        assert list(storage) == ["hello"]
        del storage["hello"]
        assert len(storage) == 0


if __name__ == "__main__":
    test_s3_storage()
