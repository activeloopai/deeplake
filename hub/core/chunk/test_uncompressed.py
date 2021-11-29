from hub.constants import MB, PARTIAL_NUM_SAMPLES
from hub.core.chunk.uncompressed_chunk import UncompressedChunk
import numpy as np
import pytest

import hub
from hub.core.meta.tensor_meta import TensorMeta
from hub.core.sample import Sample
from hub.core.tiling.deserialize import np_list_to_sample
from hub.core.tiling.sample_tiles import SampleTiles  # type: ignore


common_args = {
    "min_chunk_size": 16 * MB,
    "max_chunk_size": 32 * MB,
    "compression": None,
}


def create_tensor_meta():
    tensor_meta = TensorMeta()
    tensor_meta.dtype = "uint8"
    tensor_meta.max_shape = None
    tensor_meta.min_shape = None
    tensor_meta.htype = None
    tensor_meta.length = 0
    return tensor_meta


def test_read_write_sequence():
    common_args["tensor_meta"] = create_tensor_meta()
    data_in = [np.random.rand(500, 500).astype("uint8") for _ in range(10)]
    while data_in:
        chunk = UncompressedChunk(**common_args)
        num_samples = int(chunk.extend_if_has_space(data_in))
        data_out = [chunk.read_sample(i) for i in range(num_samples)]
        np.testing.assert_array_equal(data_out, data_in[:num_samples])
        data_in = data_in[num_samples:]


def test_read_write_sequence_big(cat_path):
    common_args["tensor_meta"] = create_tensor_meta()
    data_in = []
    for i in range(50):
        if i % 10 == 0:
            data_in.append(np.random.rand(3000, 3000, 3).astype("uint8"))
        elif i % 3 == 0:
            data_in.append(hub.read(cat_path))
        else:
            data_in.append(np.random.rand(500, 500, 3).astype("uint8"))
    data_in2 = data_in.copy()
    tiles = []
    original_length = len(data_in)

    while data_in:
        chunk = UncompressedChunk(**common_args)
        if isinstance(data_in[0], SampleTiles):
            sample = data_in[0]
            if sample.is_last_write:
                current_length = len(data_in)
                index = original_length - current_length
                full_data_out = np_list_to_sample(
                    tiles,
                    sample.sample_shape,
                    sample.tile_shape,
                    sample.tiles.shape,
                    "uint8",
                )
                np.testing.assert_array_equal(full_data_out, data_in2[index])
                data_in = data_in[1:]
                tiles = []

        num_samples = chunk.extend_if_has_space(data_in)
        if num_samples == PARTIAL_NUM_SAMPLES:
            tiles.append(chunk.read_sample(0))
        elif num_samples > 0:
            data_out = [chunk.read_sample(i) for i in range(num_samples)]
            for i, item in enumerate(data_out):
                if isinstance(item, Sample):
                    item = item.array
                np.testing.assert_array_equal(item, data_in[i])
            data_in = data_in[num_samples:]

def test_read_write_numpy():
    common_args["tensor_meta"] = create_tensor_meta()
    data_in = np.random.rand(10, 500, 500).astype("uint8")
    while len(data_in) > 0:
        chunk = UncompressedChunk(**common_args)
        num_samples = int(chunk.extend_if_has_space(data_in))
        data_out = np.array([chunk.read_sample(i) for i in range(num_samples)])
        if num_samples > 0:
            np.testing.assert_array_equal(data_out, data_in[:num_samples])
        data_in = data_in[num_samples:]


def test_read_write_numpy_big():
    common_args["tensor_meta"] = create_tensor_meta()
    data_in = np.random.rand(2, 3000, 3000, 3).astype("uint8")
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
    common_args["tensor_meta"] = create_tensor_meta()
    data_in = np.random.rand(10, 500, 500).astype("uint8")
    chunk = UncompressedChunk(**common_args)
    chunk.extend_if_has_space(data_in)

    data_out = np.array([chunk.read_sample(i) for i in range(10)])
    np.testing.assert_array_equal(data_out, data_in)

    chunk.update_sample(3, np.random.rand(700, 700).astype("uint8"))
    chunk.update_sample(5, np.random.rand(3000, 3000).astype("uint8"))
    for i in range(10):
        if i == 3:
            np.testing.assert_array_equal(
                chunk.read_sample(i), np.random.rand(700, 700).astype("uint8")
            )
        elif i == 5:
            np.testing.assert_array_equal(
                chunk.read_sample(i), np.random.rand(3000, 3000).astype("uint8")
            )
        else:
            np.testing.assert_array_equal(chunk.read_sample(i), data_in[i])

