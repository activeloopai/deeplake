from hub.core.serialize import (
    encode_chunk,
    decode_chunk,
    encode_chunkids,
    decode_chunkids,
)
import numpy as np
import ctypes
import hub


def test_chunk_encoding():
    version = hub.__version__
    shape_info = np.cast[hub.constants.ENCODING_DTYPE](
        np.random.randint(100, size=(17, 63))
    )
    byte_positions = np.cast[hub.constants.ENCODING_DTYPE](
        np.random.randint(100, size=(31, 3))
    )
    data = [b"1234" * 7, b"abcdefg" * 8, b"qwertyuiop" * 9]
    encoded = bytes(encode_chunk(version, shape_info, byte_positions, data))

    # from bytes
    decoded = decode_chunk(encoded)
    version2, shape_info2, byte_positions2, data2 = decoded
    assert version2 == version
    np.testing.assert_array_equal(shape_info, shape_info2)
    np.testing.assert_array_equal(byte_positions, byte_positions2)
    assert b"".join(data) == bytes(data2)


def test_chunkids_encoding():
    version = hub.__version__
    shards = [
        np.cast[hub.constants.ENCODING_DTYPE](np.random.randint(100, size=(100, 2)))
    ]
    encoded = encode_chunkids(version, shards)
    decoded = decode_chunkids(encoded)
    version2, ids = decoded
    assert version2 == version
    np.testing.assert_array_equal(np.concatenate(shards), ids)
