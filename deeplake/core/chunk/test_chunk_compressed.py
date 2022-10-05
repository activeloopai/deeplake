from deeplake.constants import MB, KB, PARTIAL_NUM_SAMPLES
from deeplake.core.chunk.chunk_compressed_chunk import ChunkCompressedChunk
import numpy as np
import pytest

import deeplake
from deeplake.core.meta.tensor_meta import TensorMeta
from deeplake.core.sample import Sample  # type: ignore
from deeplake.core.tiling.deserialize import np_list_to_sample
from deeplake.core.tiling.sample_tiles import SampleTiles

compressions_paremetrized = pytest.mark.parametrize("compression", ["lz4", "png"])

common_args = {
    "min_chunk_size": 1 * MB,
    "max_chunk_size": 2 * MB,
    "tiling_threshold": 1 * MB,
}


def create_tensor_meta():
    tensor_meta = TensorMeta()
    tensor_meta.dtype = "uint8"
    tensor_meta.max_shape = None
    tensor_meta.min_shape = None
    tensor_meta.htype = "generic"
    tensor_meta.length = 0
    return tensor_meta


@compressions_paremetrized
def test_read_write_sequence(compression):
    tensor_meta = create_tensor_meta()
    common_args["tensor_meta"] = tensor_meta
    common_args["compression"] = compression
    dtype = tensor_meta.dtype
    data_in = [
        np.random.randint(0, 255, size=(250, 125)).astype(dtype) for _ in range(10)
    ]
    data_in2 = data_in.copy()
    while data_in:
        chunk = ChunkCompressedChunk(**common_args)
        num_samples = int(chunk.extend_if_has_space(data_in))
        chunk._decompressed_samples = None
        data_out = [chunk.read_sample(i) for i in range(num_samples)]
        np.testing.assert_array_equal(data_out, data_in2[:num_samples])
        data_in = data_in[num_samples:]
        data_in2 = data_in2[num_samples:]


@compressions_paremetrized
@pytest.mark.parametrize("random", [True, False])
def test_read_write_sequence_big(cat_path, compression, random):
    tensor_meta = create_tensor_meta()
    common_args["tensor_meta"] = tensor_meta
    common_args["compression"] = compression
    dtype = tensor_meta.dtype
    data_in = []
    for i in range(50):
        if i % 10 == 0:
            data_in.append(
                np.random.randint(0, 255, size=(1501, 750, 3)).astype(dtype) * random
            )
        elif i % 3 == 0:
            data_in.append(
                deeplake.read(cat_path)
                if random
                else np.zeros((225, 225, 3), dtype=dtype)
            )
        else:
            data_in.append(
                np.random.randint(0, 255, size=(250, 125, 3)).astype(dtype) * random
            )
    data_in2 = data_in.copy()
    tiles = []
    original_length = len(data_in)
    tiled = False
    while data_in:
        chunk = ChunkCompressedChunk(**common_args)
        chunk._compression_ratio = 10  # start with a bad compression ratio to hit exponential back off code path
        num_samples = chunk.extend_if_has_space(data_in)
        if num_samples == PARTIAL_NUM_SAMPLES:
            tiled = True
            tiles.append(chunk.read_sample(0, is_tile=True))
            sample = data_in[0]
            assert isinstance(sample, SampleTiles)
            if sample.is_last_write:
                current_length = len(data_in)
                index = original_length - current_length
                full_data_out = np_list_to_sample(
                    tiles,
                    sample.sample_shape,
                    sample.tile_shape,
                    sample.layout_shape,
                    dtype,
                )
                np.testing.assert_array_equal(full_data_out, data_in2[index])
                data_in = data_in[1:]
                tiles = []

        elif num_samples > 0:
            data_out = [chunk.read_sample(i) for i in range(num_samples)]
            for i, item in enumerate(data_out):
                if isinstance(item, Sample):
                    item = item.array
                np.testing.assert_array_equal(item, data_in[i])
            data_in = data_in[num_samples:]
    assert tiled


@compressions_paremetrized
def test_update(compression):
    tensor_meta = create_tensor_meta()
    common_args["tensor_meta"] = tensor_meta
    common_args["compression"] = compression
    dtype = tensor_meta.dtype
    arr = np.random.randint(0, 255, size=(7, 75, 50, 3)).astype(dtype)
    data_in = list(arr)
    chunk = ChunkCompressedChunk(**common_args)
    chunk.extend_if_has_space(data_in)

    data_out = np.array([chunk.read_sample(i) for i in range(7)])
    np.testing.assert_array_equal(data_out, data_in)

    data_3 = np.random.randint(0, 255, size=(175, 350, 3)).astype(dtype)
    data_5 = np.random.randint(0, 255, size=(500, 750, 3)).astype(dtype)

    chunk.update_sample(3, data_3)
    chunk.update_sample(5, data_5)
    for i in range(7):
        if i == 3:
            np.testing.assert_array_equal(chunk.read_sample(i), data_3)
        elif i == 5:
            np.testing.assert_array_equal(chunk.read_sample(i), data_5)
        else:
            np.testing.assert_array_equal(chunk.read_sample(i), data_in[i])
