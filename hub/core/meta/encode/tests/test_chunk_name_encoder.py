import pytest
import numpy as np
from hub.core.meta.encode.chunk_name import ChunkNameEncoder
from hub.core.storage.provider import StorageProvider


def test_trivial():
    enc = ChunkNameEncoder()

    enc.append_chunk(10)

    id1 = enc[0]
    assert enc[9] == id1

    enc.extend_chunk(10)
    enc.extend_chunk(9)
    enc.extend_chunk(1)

    # new chunks
    enc.append_chunk(1)
    enc.append_chunk(5)

    enc.extend_chunk(1)

    id2 = enc[30]
    id3 = enc[31]

    assert id1 != id2
    assert id2 != id3
    assert id1 != id3

    assert enc[10] == id1
    assert enc[29] == id1
    assert enc[35] == id3
    assert enc[36] == id3

    assert enc.num_samples == 37
    assert len(enc._encoded) == 3


def test_auto_combine():
    enc = ChunkNameEncoder()

    # these can all be squeezed into 1 chunk id
    enc.append_chunk(10)
    enc.append_chunk(10)
    enc.append_chunk(10)
    enc.append_chunk(5)

    # cannot combine yet
    assert enc[30] != enc[20]

    # now can combine
    enc.extend_chunk(5)

    assert enc[0] == enc[10]
    assert enc[10] == enc[20]
    assert enc[30] == enc[35]
    assert enc[0] == enc[35]

    assert enc.num_samples == 40

    # should be 1 because chunks with the same counts can be combined
    assert len(enc._encoded) == 1

    enc.append_chunk(9)

    # cannot combine
    assert len(enc._encoded) == 2

    enc.append_chunk(10)

    # cannot combine
    assert len(enc._encoded) == 3

    enc.append_chunk(3)

    # cannot combine
    assert len(enc._encoded) == 4

    enc.extend_chunk(7)

    # now can combine
    assert len(enc._encoded) == 3


def test_failures():
    enc = ChunkNameEncoder()

    # cannot extend previous if no samples exist
    with pytest.raises(Exception):  # TODO: exceptions.py
        enc.extend_chunk(0)

    with pytest.raises(IndexError):
        enc[-1]

    enc.append_chunk(10)

    with pytest.raises(ValueError):
        enc.extend_chunk(0)

    with pytest.raises(ValueError):
        enc.append_chunk(0)

    with pytest.raises(IndexError):
        enc[10]
