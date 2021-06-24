import pytest
from hub.core.meta.encode.chunk_name import ChunkNameEncoder


def _assert_valid_encodings(enc: ChunkNameEncoder):
    assert len(enc._encoded) == len(enc._connectivity)


def test_trivial():
    enc = ChunkNameEncoder()

    enc.append_chunk(10)

    # assert not enc.try_combining_last_two_chunks()

    id1 = enc.get_chunk_names(0)
    assert enc.get_chunk_names(9) == id1

    enc.extend_chunk(10)
    enc.extend_chunk(9)
    enc.extend_chunk(1)

    # new chunks
    enc.append_chunk(1)
    enc.append_chunk(5)

    enc.extend_chunk(1)

    id2 = enc.get_chunk_names(30)
    id3 = enc.get_chunk_names(31)

    assert id1 != id2
    assert id2 != id3
    assert id1 != id3

    assert enc.get_chunk_names(10) == id1
    assert enc.get_chunk_names(29) == id1
    assert enc.get_chunk_names(35) == id3
    assert enc.get_chunk_names(36) == id3

    # assert not enc.try_combining_last_two_chunks()

    assert enc.num_samples == 37
    assert len(enc._encoded) == 3

    # make sure the length of all returned chunk names = 1
    for i in range(0, 37):
        assert len(enc.get_chunk_names(i)) == 1

    _assert_valid_encodings(enc)


def test_multi_chunks_per_sample():
    # TODO:
    enc = ChunkNameEncoder()

    # idx=0:5 samples fit in chunk 0
    # idx=5 sample fits in chunk 0, chunk 1, chunk 2, and chunk 3
    enc.append_chunk(1)
    enc.extend_chunk(5, connected_to_next=True)
    enc.append_chunk(0, connected_to_next=True)  # continuation of the 6th sample
    enc.append_chunk(0, connected_to_next=True)  # continuation of the 6th sample
    enc.append_chunk(0, connected_to_next=False)  # end of the 6th sample

    enc.extend_chunk(3)  # these samples are part of the last chunk

    enc.append_chunk(10_000)
    enc.extend_chunk(10)

    assert len(enc.get_chunk_names(0)) == 1
    assert len(enc.get_chunk_names(4)) == 1
    assert enc.get_chunk_names(0) == enc.get_chunk_names(4)
    s5_chunks = enc.get_chunk_names(5)
    assert len(s5_chunks) == 4
    assert len(set(s5_chunks)) == len(s5_chunks)

    assert len(enc.get_chunk_names(6)) == 1
    assert len(enc.get_chunk_names(7)) == 1
    assert len(enc.get_chunk_names(8)) == 1

    assert s5_chunks[-1] == enc.get_chunk_names(6)[0]
    assert s5_chunks[-1] == enc.get_chunk_names(6)[0]

    assert enc.num_samples == 10_019

    # sample takes up 2 chunks
    enc.append_chunk(1, connected_to_next=True)
    enc.append_chunk(0, connected_to_next=False)

    assert enc.num_samples == 10_020
    assert len(enc.get_chunk_names(10_019)) == 2

    # sample takes up 5 chunks
    enc.append_chunk(1, connected_to_next=True)
    enc.append_chunk(0, connected_to_next=True)
    enc.append_chunk(0, connected_to_next=True)
    enc.append_chunk(0, connected_to_next=True)
    enc.append_chunk(0, connected_to_next=False)

    assert len(enc.get_chunk_names(10_020)) == 5
    assert enc.num_samples == 10_021

    _assert_valid_encodings(enc)


def test_failures():
    enc = ChunkNameEncoder()

    with pytest.raises(Exception):
        # fails because no previous chunk exists
        enc.append_chunk(0)

    # cannot extend previous if no samples exist
    with pytest.raises(Exception):  # TODO: exceptions.py
        enc.extend_chunk(1)

    with pytest.raises(IndexError):
        enc.get_chunk_names(-1)

    enc.append_chunk(10)
    enc.extend_chunk(10, connected_to_next=True)

    with pytest.raises(Exception):
        enc.extend_chunk(1)  # cannot extend because already connected to next

    with pytest.raises(ValueError):
        enc.extend_chunk(0)  # not allowed

    with pytest.raises(ValueError):
        enc.extend_chunk(-1)

    enc.append_chunk(0)  # this is allowed

    enc.append_chunk(1, connected_to_next=True)
    enc.append_chunk(0, connected_to_next=True)

    with pytest.raises(Exception):
        enc.extend_chunk(1)

    with pytest.raises(ValueError):
        enc.append_chunk(-1)

    enc.append_chunk(0, connected_to_next=False)  # end this sample

    with pytest.raises(Exception):
        # fails because previous chunk is not connected to next
        enc.append_chunk(0)

    with pytest.raises(IndexError):
        enc.get_chunk_names(21)

    _assert_valid_encodings(enc)
