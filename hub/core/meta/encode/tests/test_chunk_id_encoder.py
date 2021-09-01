from hub.util.chunks import chunk_id_from_name, chunk_name_from_id, derive_tile_shape, is_tile_chunk_id, random_chunk_id
from hub.constants import ENCODING_DTYPE
from hub.util.exceptions import ChunkIdEncoderError
import pytest
from hub.core.meta.encode.chunk_id import (
    ChunkIdEncoder,
)


def test_trivial():
    enc = ChunkIdEncoder()

    assert enc.num_chunks == 0

    id1 = enc.generate_chunk_id()
    enc.register_samples(10)

    assert enc.num_chunks == 1

    assert id1 == enc[0]
    assert id1 == enc[9]

    enc.register_samples(10)
    enc.register_samples(9)
    enc.register_samples(1)

    assert enc.num_chunks == 1
    assert enc.num_samples == 30

    assert id1 == enc[10]
    assert id1 == enc[11]
    assert id1 == enc[29]

    # new chunks
    id2 = enc.generate_chunk_id()
    enc.register_samples(1)

    id3 = enc.generate_chunk_id()
    enc.register_samples(5)

    assert enc.num_chunks == 3
    assert enc.num_samples == 36

    assert id1 != id2
    assert id2 != id3

    assert id1 == enc[29]
    assert id2 == enc[30]
    assert id3 == enc[31]

    # test local indexing
    assert enc.translate_index_relative_to_chunks(0) == 0
    assert enc.translate_index_relative_to_chunks(1) == 1
    assert enc.translate_index_relative_to_chunks(29) == 29
    assert enc.translate_index_relative_to_chunks(30) == 0
    assert enc.translate_index_relative_to_chunks(31) == 0
    assert enc.translate_index_relative_to_chunks(35) == 4


def test_tiles():
    enc = ChunkIdEncoder()

    assert enc.num_chunks == 0

    id0 = enc.generate_chunk_id()
    enc.register_samples(1)

    assert id0 == enc[0]

    id1 = enc.generate_chunk_id()
    enc.register_samples(1)

    assert id1 == enc[1]

    id2 = enc.generate_chunk_id()
    enc.register_samples(0)

    assert [id1, id2] == enc[1]

    id3 = enc.generate_chunk_id()
    enc.register_samples(0)

    assert [id1, id2, id3] == enc[1]

    with pytest.raises(IndexError):
        enc[2]

    id4 = enc.generate_chunk_id()
    enc.register_samples(1)

    assert id0 == enc[0]
    assert [id1, id2, id3] == enc[1]
    assert id4 == enc[2]

    assert enc.num_chunks == 5
    assert enc.num_samples == 3


def test_failures():
    enc = ChunkIdEncoder()

    with pytest.raises(ChunkIdEncoderError):
        # fails because no chunk ids exist
        enc.register_samples(0)

    enc.generate_chunk_id()

    enc.register_samples(1)

    with pytest.raises(IndexError):
        enc[1]

    enc.generate_chunk_id()

    with pytest.raises(IndexError):
        enc[1]


def test_ids():
    enc = ChunkIdEncoder()

    id = enc.generate_chunk_id()
    assert id.itemsize == ENCODING_DTYPE(1).itemsize
    name = chunk_name_from_id(id)
    out_id = chunk_id_from_name(name)

    assert id == out_id


def test_tile_ids():
    enc = ChunkIdEncoder()

    # generate 2 tile chunks
    root_id = random_chunk_id()
    tile_id0 = enc.generate_tile_chunk_id(root_id, (1, 2, 3))
    enc.register_samples(1)
    tile_id1 = enc.generate_tile_chunk_id(root_id, (4, 5, 6, 7))
    enc.register_samples(0)

    assert enc[0] == [tile_id0, tile_id1]

    assert not is_tile_chunk_id(root_id)
    assert is_tile_chunk_id(tile_id0)
    assert is_tile_chunk_id(tile_id1)

    assert derive_tile_shape(tile_id0) == (1, 2, 3)
    assert derive_tile_shape(tile_id1) == (4, 5, 6, 7)

    # cannot add more samples to a tile chunk
    with pytest.raises(NotImplementedError):    # TODO: exceptions.py
        enc.register_samples(0)
    with pytest.raises(NotImplementedError):    # TODO: exceptions.py
        enc.register_samples(1)

    assert enc.num_samples == 1
    assert enc.num_chunks == 2

    # generate some normal chunks
    id2 = enc.generate_chunk_id()
    enc.register_samples(10)

    id3 = enc.generate_chunk_id()
    enc.register_samples(1)

    assert enc.num_samples == 12
    assert enc.num_chunks == 4

    assert enc[0] == [tile_id0, tile_id1]
    assert enc[1] == id2
    assert enc[2] == id3