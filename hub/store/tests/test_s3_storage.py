from concurrent.futures.thread import ThreadPoolExecutor

import boto3
import pytest
from s3fs import S3FileSystem
import cloudpickle
import numpy as np

from hub.store.s3_storage import S3Storage
from hub.utils import s3_creds_exist


def create_client():
    return boto3.client("s3")


BYTE_DATA = bytes("world2", "utf-8")
NUMPY_ARR = np.array([[4, 512, 512]])


@pytest.mark.skipif(not s3_creds_exist(), reason="Requires s3 credentials")
def test_s3_storage():
    _storage = S3FileSystem()
    storage = S3Storage(_storage, "s3://snark-test/test_s3_storage1")

    storage["hello"] = BYTE_DATA
    storage["numpy"] = NUMPY_ARR
    assert storage["hello"] == BYTE_DATA
    assert storage["numpy"] == bytearray(memoryview(NUMPY_ARR))
    assert len(storage) == 2
    assert list(storage) == ["hello", "numpy"]
    del storage["hello"]
    assert len(storage) == 1


@pytest.mark.skipif(not s3_creds_exist(), reason="Requires s3 credentials")
def test_s3_storage_pickability():
    _storage = S3FileSystem()
    storage = S3Storage(_storage, "s3://snark-test/test_s3_storage")

    cloudpickle.dumps(storage)


if __name__ == "__main__":
    test_s3_storage()
