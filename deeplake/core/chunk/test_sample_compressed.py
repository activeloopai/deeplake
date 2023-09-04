from deeplake.constants import MB, PARTIAL_NUM_SAMPLES
from deeplake.core.chunk.sample_compressed_chunk import SampleCompressedChunk
import numpy as np
import pytest

import deeplake
from deeplake.core.meta.tensor_meta import TensorMeta
from deeplake.core.sample import Sample  # type: ignore
from deeplake.core.tiling.deserialize import np_list_to_sample
from deeplake.core.tiling.sample_tiles import SampleTiles

compressions_paremetrized = pytest.mark.parametrize("compression", ["lz4"])


common_args = {
    "min_chunk_size": 1 * MB,
    "max_chunk_size": 2 * MB,
    "tiling_threshold": 1 * MB,
}


def create_tensor_meta():
    tensor_meta = TensorMeta()
    tensor_meta.dtype = "float32"
    tensor_meta.max_shape = None
    tensor_meta.min_shape = None
    tensor_meta.htype = None
    tensor_meta.length = 0
    return tensor_meta


@compressions_paremetrized
def test_read_write_sequence(compression):
    tensor_meta = create_tensor_meta()
    common_args["tensor_meta"] = tensor_meta
    common_args["compression"] = compression
    dtype = tensor_meta.dtype
    data_in = [np.random.rand(250, 125, 3).astype(dtype) for _ in range(10)]
    data_in2 = data_in.copy()
    while data_in:
        chunk = SampleCompressedChunk(**common_args)
        num_samples = int(chunk.extend_if_has_space(data_in))
        data_out = [chunk.read_sample(i) for i in range(num_samples)]
        np.testing.assert_array_equal(data_out, data_in2[:num_samples])
        data_in = data_in[num_samples:]
        data_in2 = data_in2[num_samples:]


@pytest.mark.slow
@compressions_paremetrized
def test_read_write_sequence_big(cat_path, compression):
    tensor_meta = create_tensor_meta()
    common_args = {
        "min_chunk_size": 16 * MB,
        "max_chunk_size": 32 * MB,
        "tiling_threshold": 16 * MB,
        "tensor_meta": tensor_meta,
        "compression": compression,
    }

    dtype = tensor_meta.dtype
    data_in = []
    for i in range(50):
        if i % 10 == 0:
            data_in.append(np.random.rand(6001, 3000, 3).astype(dtype))
        elif i % 3 == 0:
            data_in.append(deeplake.read(cat_path))
        else:
            data_in.append(np.random.rand(1000, 500, 3).astype(dtype))
    data_in2 = data_in.copy()
    tiles = []
    original_length = len(data_in)

    while data_in:
        chunk = SampleCompressedChunk(**common_args)
        num_samples = chunk.extend_if_has_space(data_in)
        if num_samples == PARTIAL_NUM_SAMPLES:
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


# @compressions_paremetrized
# def test_update(compression):
#     tensor_meta = create_tensor_meta()
#     common_args["tensor_meta"] = tensor_meta
#     common_args["compression"] = compression
#     dtype = tensor_meta.dtype
#     arr = np.random.rand(7, 25, 125, 3).astype(dtype)
#     data_in = list(arr)
#     chunk = SampleCompressedChunk(**common_args)
#     chunk.extend_if_has_space(data_in)
#     data_out = np.array([chunk.read_sample(i) for i in range(7)])
#     np.testing.assert_array_equal(data_out, data_in)

#     data_3 = np.random.rand(175, 350, 3).astype(dtype)
#     data_5 = np.random.rand(1500, 750, 3).astype(dtype)

#     chunk.update_sample(3, data_3)
#     chunk.update_sample(5, data_5)
#     for i in range(7):
#         if i == 3:
#             np.testing.assert_array_equal(chunk.read_sample(i), data_3)
#         elif i == 5:
#             np.testing.assert_array_equal(chunk.read_sample(i), data_5)
#         else:
#             np.testing.assert_array_equal(chunk.read_sample(i), arr[i])
