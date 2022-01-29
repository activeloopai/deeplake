from time import time
from numpy import (
    cast as np_cast,
    random as np_random,
    concatenate as np_concatenate,
    testing as np_testing
)

from hub import __version__
from hub.constants import ENCODING_DTYPE
from hub.core.serialize import (
    serialize_chunk,
    deserialize_chunk,
    serialize_chunkids,
    deserialize_chunkids,
)


def test_chunk_serialize():
    version = __version__
    shape_info = np_cast[ENCODING_DTYPE](
        np_random.randint(100, size=(17, 63))
    )
    byte_positions = np_cast[ENCODING_DTYPE](
        np_random.randint(100, size=(31, 3))
    )
    data = [b"x" * 8 * 1024 * 1024] * 2  # 16 MB chunk
    encoded = bytes(serialize_chunk(version, shape_info, byte_positions, data))

    # Deserialize
    start = time()
    for _ in range(10000):
        decoded = deserialize_chunk(encoded)
    end = time()

    assert end - start < 0.5

    version2, shape_info2, byte_positions2, data2 = decoded
    assert version2 == version
    np_testing.assert_array_equal(shape_info, shape_info2)
    np_testing.assert_array_equal(byte_positions, byte_positions2)
    assert b"".join(data) == bytes(data2)


def test_chunkids_serialize():
    version = __version__
    shards = [
        np_cast[ENCODING_DTYPE](np_random.randint(100, size=(100, 2)))
    ]
    encoded = serialize_chunkids(version, shards)
    decoded = deserialize_chunkids(encoded)
    version2, ids = decoded
    assert version2 == version
    np_testing.assert_array_equal(np_concatenate(shards), ids)
