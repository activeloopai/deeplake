from hub.constants import MB, PARTIAL_NUM_SAMPLES
from hub.core.chunk.uncompressed_chunk import UncompressedChunk
import numpy as np
import pytest

import hub
from hub.core.meta.tensor_meta import TensorMeta
from hub.core.sample import Sample  # type: ignore
from hub.core.tiling.deserialize import np_list_to_sample
from hub.core.tiling.sample_tiles import SampleTiles


common_args = {
    "min_chunk_size": 1 * MB,
    "max_chunk_size": 2 * MB,
    "tiling_threshold": 1 * MB,
    "compression": None,
}


def create_tensor_meta():
    tensor_meta = TensorMeta()
    tensor_meta.dtype = "float64"
    tensor_meta.max_shape = None
    tensor_meta.min_shape = None
    tensor_meta.htype = None
    tensor_meta.length = 0
    return tensor_meta


def test_read_write_sequence():
    tensor_meta = create_tensor_meta()
    common_args["tensor_meta"] = tensor_meta
    dtype = tensor_meta.dtype
    data_in = [np.random.rand(125, 125).astype(dtype) for _ in range(10)]
    while data_in:
        chunk = UncompressedChunk(**common_args)
        num_samples = int(chunk.extend_if_has_space(data_in))
        data_out = [chunk.read_sample(i) for i in range(num_samples)]
        np.testing.assert_array_equal(data_out, data_in[:num_samples])
        data_in = data_in[num_samples:]


def test_read_write_sequence_big(cat_path):
    tensor_meta = create_tensor_meta()
    common_args["tensor_meta"] = tensor_meta
    dtype = tensor_meta.dtype
    data_in = []
    for i in range(50):
        if i % 10 == 0:
            data_in.append(np.random.rand(751, 750, 3).astype(dtype))
        elif i % 3 == 0:
            data_in.append(hub.read(cat_path))
        else:
            data_in.append(np.random.rand(125, 125, 3).astype(dtype))
    data_in2 = data_in.copy()
    tiles = []
    original_length = len(data_in)

    while data_in:
        chunk = UncompressedChunk(**common_args)
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


def test_read_write_numpy():
    tensor_meta = create_tensor_meta()
    common_args["tensor_meta"] = tensor_meta
    dtype = tensor_meta.dtype
    data_in = np.random.rand(10, 125, 125).astype(dtype)
    while len(data_in) > 0:
        chunk = UncompressedChunk(**common_args)
        num_samples = int(chunk.extend_if_has_space(data_in))
        data_out = np.array([chunk.read_sample(i) for i in range(num_samples)])
        if num_samples > 0:
            np.testing.assert_array_equal(data_out, data_in[:num_samples])
        data_in = data_in[num_samples:]


def test_read_write_numpy_big():
    tensor_meta = create_tensor_meta()
    common_args["tensor_meta"] = tensor_meta
    dtype = tensor_meta.dtype
    data_in = np.random.rand(2, 750, 750, 3).astype(dtype)
    prev_num_samples = None
    with pytest.raises(ValueError):
        while len(data_in) > 0:
            chunk = UncompressedChunk(**common_args)
            num_samples = int(chunk.extend_if_has_space(data_in))
            if num_samples == 0 and prev_num_samples == 0:
                raise ValueError(
                    "Unexpected, bigger numpy arrays should be sent as sequence to chunk"
                )
            data_out = np.array([chunk.read_sample(i) for i in range(num_samples)])
            if num_samples > 0:
                np.testing.assert_array_equal(data_out, data_in[:num_samples])
            data_in = data_in[num_samples:]
            prev_num_samples = num_samples


def test_update():
    tensor_meta = create_tensor_meta()
    common_args["tensor_meta"] = tensor_meta
    dtype = tensor_meta.dtype
    data_in = np.random.rand(7, 125, 125).astype(dtype)
    chunk = UncompressedChunk(**common_args)
    chunk.extend_if_has_space(data_in)

    data_out = np.array([chunk.read_sample(i) for i in range(7)])
    np.testing.assert_array_equal(data_out, data_in)

    data_3 = np.random.rand(175, 175).astype(dtype)
    data_5 = np.random.rand(375, 375).astype(dtype)

    chunk.update_sample(3, data_3)
    chunk.update_sample(5, data_5)
    for i in range(7):
        if i == 3:
            np.testing.assert_array_equal(chunk.read_sample(i), data_3)
        elif i == 5:
            np.testing.assert_array_equal(chunk.read_sample(i), data_5)
        else:
            np.testing.assert_array_equal(chunk.read_sample(i), data_in[i])
