from typing import List
import pytest
from concurrent.futures import ThreadPoolExecutor, Future

import cloudpickle
from fsspec.spec import AbstractFileSystem

from hub.store.s3_file_system_replacement import S3FileSystemReplacement
from hub.utils import s3_creds_exist
from hub.defaults import MAX_CONNECTION_WORKERS

DATA = bytes("YES YES YES!", "utf-8")
KEY = "pickable"


def manage_storage(storage: AbstractFileSystem, url: str) -> None:
    if storage.exists(url):
        storage.rm(url, recursive=True)
    storage.makedirs(url, exist_ok=False)


@pytest.mark.skipif(not s3_creds_exist(), reason="Requires s3 credentials")
def test_s3_file_system_replacement_pickability(
    url: str = "snark-test/test_s3_file_system_replacement_pickability",
) -> None:
    storage = S3FileSystemReplacement()
    manage_storage(storage, url)
    fsmap = storage.get_mapper(url)
    fsmap[KEY] = DATA
    dumped_storage = cloudpickle.dumps(storage)
    loaded_storage = cloudpickle.loads(dumped_storage)
    loaded_fsmap = loaded_storage.get_mapper(url)
    assert loaded_fsmap[KEY] == DATA


@pytest.mark.skipif(not s3_creds_exist(), reason="Requires s3 credentials")
def test_s3_file_system_singlethreaded(
    url: str = "snark-test/test_s3_file_system_singlethreaded",
) -> None:
    storage = S3FileSystemReplacement()
    manage_storage(storage, url)
    fsmap = storage.get_mapper(url)
    fsmap[KEY] = DATA
    assert fsmap[KEY] == DATA
    del fsmap[KEY]
    try:
        fsmap[KEY]
    except Exception as ex:
        assert isinstance(ex, KeyError)


@pytest.mark.skipif(not s3_creds_exist(), reason="Requires s3 credentials")
def test_s3_file_system_multithreaded(
    url: str = "snark-test/test_s3_file_system_multithreaded",
) -> None:
    requests = range(8)
    keys = [f"KEY_{i}" for i in requests]
    storage = S3FileSystemReplacement()
    manage_storage(storage, url)
    fsmap = storage.get_mapper(url)
    with ThreadPoolExecutor(MAX_CONNECTION_WORKERS) as pool:
        futures: List[Future] = []
        for key in keys:
            futures.append(pool.submit(fsmap.__setitem__, key, DATA))
        for f in futures:
            assert f.result() is None
        items = pool.map(fsmap.__getitem__, keys)
        for item in items:
            assert item == DATA
        items = pool.map(fsmap.__delitem__, keys)
        for item in items:
            assert item == None
        futures = []
        for key in keys:
            futures.append(pool.submit(fsmap.__getitem__, key))
        for f in futures:
            try:
                f.result()
            except Exception as ex:
                assert isinstance(ex, KeyError)
