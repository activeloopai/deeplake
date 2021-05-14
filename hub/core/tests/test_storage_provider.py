import pytest

from hub.core.tests.common import parametrize_all_storage_providers
from hub.util.cache_chain import get_cache_chain
import pytest
from hub.util.s3 import has_s3_credentials

NUM_FILES = 20
MB = 1024 * 1024


@parametrize_all_storage_providers
def test_storage_provider(storage):
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
    storage.flush()
