from hub.core.storage import MemoryProvider, LocalProvider, S3Provider
from hub.core.storage.mapped_provider import MappedProvider
import pytest
from hub.util.check_s3_creds import s3_creds_exist


def test_provider(provider=MappedProvider()):
    FILE_1 = "abc.txt"
    FILE_2 = "def.txt"

    provider[FILE_1] = b"hello world"
    assert provider[FILE_1] == b"hello world"
    assert provider.__getitem__(FILE_1, 2, 5) == b"llo"

    provider.__setitem__(FILE_1, b"abcde", 6)
    assert provider[FILE_1] == b"hello abcde"

    provider.__setitem__(FILE_1, b"tuvwxyz", 6)
    assert provider[FILE_1] == b"hello tuvwxyz"

    provider.__setitem__(FILE_2, b"hello world", 3)
    assert provider[FILE_2] == b"\x00\x00\x00hello world"
    provider.__setitem__(FILE_2, b"new_text", overwrite=True)
    assert provider[FILE_2] == b"new_text"

    assert len(provider) >= 1

    for _ in provider:
        pass

    del provider[FILE_1]
    del provider[FILE_2]

    with pytest.raises(KeyError):
        provider[FILE_1]


def test_memory_provider():
    test_provider(MemoryProvider("abc/def"))


def test_local_provider():
    test_provider(LocalProvider("./hub_storage_local_test"))


@pytest.mark.skipif(not s3_creds_exist(), reason="requires s3 credentials")
def test_s3_provider():
    test_provider(S3Provider("snark-test/hub_storage_s3_test"))


# TODO add pytest benchmarks
