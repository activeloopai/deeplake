import pytest
import numpy as np
from hub.core.meta.encode.chunk_name import ChunkNameEncoder
from hub.core.storage.provider import StorageProvider


def test_trivial():
    enc = ChunkNameEncoder()

    enc.append_chunk(10)

    assert not enc.try_combining_last_two_chunks()

    id1 = enc.get_chunk_id(0)
    assert enc.get_chunk_id(9) == id1

    enc.extend_chunk(10)
    enc.extend_chunk(9)
    enc.extend_chunk(1)

    # new chunks
    enc.append_chunk(1)
    enc.append_chunk(5)

    enc.extend_chunk(1)

    id2 = enc.get_chunk_id(30)
    id3 = enc.get_chunk_id(31)

    assert id1 != id2
    assert id2 != id3
    assert id1 != id3

    assert enc.get_chunk_id(10) == id1
    assert enc.get_chunk_id(29) == id1
    assert enc.get_chunk_id(35) == id3
    assert enc.get_chunk_id(36) == id3

    assert not enc.try_combining_last_two_chunks()

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
    assert enc.get_chunk_id(30) != enc.get_chunk_id(20)

    # now can combine
    enc.extend_chunk(5)
    assert enc.try_combining_last_two_chunks()
    assert not enc.try_combining_last_two_chunks()  # cannot combine twice in a row

    assert enc.get_chunk_id(0) == enc.get_chunk_id(10)
    assert enc.get_chunk_id(10) == enc.get_chunk_id(20)
    assert enc.get_chunk_id(30) == enc.get_chunk_id(35)
    assert enc.get_chunk_id(0) == enc.get_chunk_id(35)

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
    assert enc.try_combining_last_two_chunks()
    assert not enc.try_combining_last_two_chunks()  # cannot combine twice in a row

    assert len(enc._encoded) == 3


def test_failures():
    enc = ChunkNameEncoder()

    with pytest.raises(Exception):  # TODO: exceptions.py
        enc.try_combining_last_two_chunks()

    # cannot extend previous if no samples exist
    with pytest.raises(Exception):  # TODO: exceptions.py
        enc.extend_chunk(0)

    with pytest.raises(IndexError):
        enc.get_chunk_id(-1)

    enc.append_chunk(10)

    with pytest.raises(ValueError):
        enc.extend_chunk(0)

    with pytest.raises(ValueError):
        enc.append_chunk(0)

    with pytest.raises(IndexError):
        enc.get_chunk_id(10)
