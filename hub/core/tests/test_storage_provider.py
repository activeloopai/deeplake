from hub.core.storage import MemoryProvider, LocalProvider, S3Provider
from hub.core.storage.mapped_provider import MappedProvider
import pytest
from hub.util.check_s3_creds import s3_creds_exist


def run_provider_test(storage):
    FILE_1 = "abc.txt"
    FILE_2 = "def.txt"

    storage[FILE_1] = b"hello world"
    assert storage[FILE_1] == b"hello world"
    assert storage.__getitem__(FILE_1, 2, 5) == b"llo"

    storage.__setitem__(FILE_1, b"abcde", 6)
    assert storage[FILE_1] == b"hello abcde"

    storage.__setitem__(FILE_1, b"tuvwxyz", 6)
    assert storage[FILE_1] == b"hello tuvwxyz"

    storage.__setitem__(FILE_2, b"hello world", 3)
    assert storage[FILE_2] == b"\x00\x00\x00hello world"
    storage.__setitem__(FILE_2, b"new_text", overwrite=True)
    assert storage[FILE_2] == b"new_text"

    assert len(storage) >= 1

    for _ in storage:
        pass

    del storage[FILE_1]
    del storage[FILE_2]

    with pytest.raises(KeyError):
        storage[FILE_1]


def test_memory_provider(memory_storage):
    run_provider_test(memory_storage)


def test_local_provider(local_storage):
    run_provider_test(local_storage)


def test_s3_provider(s3_storage):
    run_provider_test(S3Provider("snark-test/hub_storage_s3_test"))


# TODO add pytest benchmarks
