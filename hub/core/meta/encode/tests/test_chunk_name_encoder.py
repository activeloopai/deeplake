import pytest
import numpy as np
from hub.core.meta.encode.chunk_name import ChunkNameEncoder
from hub.core.storage.provider import StorageProvider


def test_trivial():
    enc = ChunkNameEncoder()

    enc.add_samples_to_chunk(10, False)

    id1 = enc[0]
    assert enc[9] == id1

    enc.add_samples_to_chunk(10, True)
    enc.add_samples_to_chunk(9, True)
    enc.add_samples_to_chunk(1, True)

    # new chunks
    enc.add_samples_to_chunk(1, False)
    enc.add_samples_to_chunk(5, False)

    enc.add_samples_to_chunk(1, True)

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


# TODO:
# def test_delimeted():
#     enc = ChunkNameEncoder()
#
#     id1 = enc.add_samples_to_chunk(10, False)
#     id2 = enc.add_samples_to_chunk(10, False)
#
#     assert id1 == id2
#
#     assert enc.num_samples == 20
#     assert len(enc._encoded) == 1  # should be 1 because id1 and id2 can be combined


def test_failures():
    enc = ChunkNameEncoder()

    # cannot extend previous if no samples exist
    with pytest.raises(Exception):  # TODO: exceptions.py
        enc.add_samples_to_chunk(0, True)

    with pytest.raises(IndexError):
        enc[-1]

    enc.add_samples_to_chunk(10, False)

    with pytest.raises(ValueError):
        enc.add_samples_to_chunk(0, True)

    with pytest.raises(ValueError):
        enc.add_samples_to_chunk(0, False)

    with pytest.raises(IndexError):
        enc[10]
