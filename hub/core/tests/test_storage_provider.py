from hub.core.storage import MemoryProvider, LocalProvider, S3BotoProvider
import pytest
from hub.core.utils import s3_creds_exist


def test_provider(provider=MemoryProvider()):
    provider["abc.txt"] = b"hello world"
    assert provider["abc.txt"] == b"hello world"
    assert provider.__getitem__("abc.txt", 2, 5) == b"llo"
    provider.__setitem__("abc.txt", b"abcde", 6)
    assert provider["abc.txt"] == b"hello abcde"
    provider.__setitem__("abc.txt", b"tuvwxyz", 6)
    assert provider["abc.txt"] == b"hello tuvwxyz"
    del provider["abc.txt"]
    with pytest.raises(KeyError):
        provider["abc.txt"]
    provider.__setitem__("def.txt", b"hello world", 3)
    assert provider["def.txt"] == b"\x00\x00\x00hello world"
    provider.__setitem__("def.txt", b"new_text", overwrite=True)
    assert provider["def.txt"] == b"new_text"
    assert len(provider) >= 1
    for _ in provider:
        pass
    del provider["def.txt"]


def test_local_provider():
    test_provider(LocalProvider("./hub_storage_local_test"))


@pytest.mark.skipif(not s3_creds_exist(), reason="requires s3 credentials")
def test_s3_provider():
    test_provider(S3BotoProvider("snark-test/hub_storage_s3_test"))

