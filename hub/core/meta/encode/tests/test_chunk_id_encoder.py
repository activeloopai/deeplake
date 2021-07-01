import pytest
from hub.core.meta.encode.chunk_id import (
    ChunkIdEncoder,
)


def _assert_valid_encodings(enc: ChunkIdEncoder):
    assert len(enc._encoded_ids) == len(enc._encoded_connectivity)


def test_trivial():
    enc = ChunkIdEncoder()

    assert enc.num_chunks == 0

    id1 = (enc.generate_chunk_id(),)
    enc.register_samples_to_last_chunk_id(10)

    assert enc.num_chunks == 1

    assert id1 == enc[0]
    assert id1 == enc[9]

    enc.register_samples_to_last_chunk_id(10)
    enc.register_samples_to_last_chunk_id(9)
    enc.register_samples_to_last_chunk_id(1)

    assert enc.num_chunks == 1
    assert enc.num_samples == 30

    assert id1 == enc[10]
    assert id1 == enc[11]
    assert id1 == enc[29]

    # new chunks
    id2 = (enc.generate_chunk_id(),)
    enc.register_samples_to_last_chunk_id(1)

    id3 = (enc.generate_chunk_id(),)
    enc.register_samples_to_last_chunk_id(5)

    assert enc.num_chunks == 3
    assert enc.num_samples == 36

    assert id1 != id2
    assert id2 != id3

    assert id1 == enc[29]
    assert id2 == enc[30]
    assert id3 == enc[31]

    _assert_valid_encodings(enc)


def assert_multi_chunks_per_sample() -> ChunkIdEncoder:
    enc = ChunkIdEncoder()

    id1 = enc.generate_chunk_id()
    enc.register_samples_to_last_chunk_id(1)

    id2 = enc.generate_chunk_id()
    enc.register_connection_to_last_chunk_id()
    enc.register_samples_to_last_chunk_id(0)

    id3 = enc.generate_chunk_id()
    enc.register_connection_to_last_chunk_id()
    enc.register_samples_to_last_chunk_id(0)

    assert (id1, id2, id3) == enc[0]
    assert enc.num_chunks == 3
    assert enc.num_samples == 1

    enc.register_samples_to_last_chunk_id(100)

    assert (id3,) == enc[1]
    assert (id3,) == enc[100]

    assert enc.num_chunks == 3
    assert enc.num_samples == 101

    return enc


def test_multi():
    assert_multi_chunks_per_sample()


def test_failures():
    enc = ChunkIdEncoder()

    with pytest.raises(Exception):
        # fails because no chunk ids exist
        enc.register_samples_to_last_chunk_id(0)

    with pytest.raises(Exception):
        # fails because no chunk ids exist
        enc.register_connection_to_last_chunk_id()

    enc.generate_chunk_id()

    with pytest.raises(Exception):
        # fails because cannot register 0 samples when there is no last chunk
        enc.register_samples_to_last_chunk_id(0)

    enc.register_samples_to_last_chunk_id(1)

    with pytest.raises(IndexError):
        enc[1]

    enc.generate_chunk_id()

    with pytest.raises(Exception):
        # cannot generate 2 chunk ids when the one before has no samples
        enc.generate_chunk_id()

    with pytest.raises(IndexError):
        enc[1]

    _assert_valid_encodings(enc)


# def test_local_indexing():
#     enc = assert_multi_chunks_per_sample()
#
#     enc.generate_chunk_id()
#     enc.register_samples_to_last_chunk_id(100)
#
#     enc.generate_chunk_id()
#     enc.register_samples_to_last_chunk_id(10)
#
#     enc.generate_chunk_id()
#     enc.register_samples_to_last_chunk_id(1000)
#
#     assert enc.num_samples == 1211
#
#     def _get_local(i: int):
#         return enc.get_local_sample_index(i)
#
#     assert _get_local(0) == 0
#     assert _get_local(5) == 5
#     assert _get_local(6) == 0
#     assert _get_local(7) == 1
#     assert _get_local(8) == 2
#     assert _get_local(9) == 0
#     assert _get_local(10) == 1
#     assert _get_local(10018) == 10009
#     assert _get_local(10019) == 0
#     assert _get_local(10020) == 0


def test_ids():
    enc = ChunkIdEncoder()

    id = enc.generate_chunk_id()
    name = ChunkIdEncoder.name_from_id(id)
    out_id = ChunkIdEncoder.id_from_name(name)

    assert id == out_id
