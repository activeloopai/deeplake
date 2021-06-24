import pytest
from hub.core.meta.encode.chunk_name import ChunkNameEncoder


def test_trivial():
    enc = ChunkNameEncoder()

    enc.append_chunk(10)

    # assert not enc.try_combining_last_two_chunks()

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

    # assert not enc.try_combining_last_two_chunks()

    assert enc.num_samples == 37
    assert len(enc._encoded) == 3


def test_multi_chunks_per_sample():
    # TODO:
    enc = ChunkNameEncoder()

    # idx=0-5 samples fit in chunk 0
    # idx=6 sample fits in chunk 0, chunk 1, chunk 2, and chunk 3
    # idx=7-10 samples fit in chunk 3

    enc.append_chunk(1)
    enc.extend_chunk(5, connected_to_next=True)
    enc.append_chunk(0)  # continuation of the 6th sample
    enc.append_chunk(0)  # continuation of the 6th sample
    enc.append_chunk(0)  # continuation of the 6th sample

    enc.extend_chunk(3)

    enc.append_chunk(10_000)
    enc.extend_chunk(10)


def test_failures():
    enc = ChunkNameEncoder()

    # cannot extend previous if no samples exist
    with pytest.raises(Exception):  # TODO: exceptions.py
        enc.extend_chunk(0)

    with pytest.raises(IndexError):
        enc.get_chunk_id(-1)

    enc.append_chunk(10)
    enc.extend_chunk(10, connected_to_next=True)

    with pytest.raises(Exception):
        enc.extend_chunk(1)  # cannot extend because already connected to next

    with pytest.raises(ValueError):
        enc.extend_chunk(0)  # not allowed

    with pytest.raises(ValueError):
        enc.extend_chunk(-1)

    enc.append_chunk(0)  # this is allowed

    enc.append_chunk(0, connected_to_next=True)

    with pytest.raises(Exception):
        enc.extend_chunk(1)

    with pytest.raises(ValueError):
        enc.append_chunk(-1)

    enc.append_chunk(0, connected_to_next=False)  # end this sample

    with pytest.raises(IndexError):
        enc.get_chunk_id(20)
