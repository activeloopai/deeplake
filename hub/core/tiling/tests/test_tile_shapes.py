import numpy as np
from hub.constants import MB
from hub.util.tiles import approximate_num_bytes, num_tiles_for_sample
from hub.core.meta.tensor_meta import TensorMeta
from hub.core.tiling.optimize import optimize_tile_shape
import pytest


# consistent simulated annealing
np.random.seed(1)


# sample_shape, sample_compression, expected_num_tiles
@pytest.mark.parametrize("config", [
    # no compression
    [(5_000, 5_000), None, 4],
    [(10_000, 10_000), None, 16],
    [(10_000, 10_000, 5), None, 64],
    [(10_000, 5_000), None, 8],
    [(100_000_000,), None, 13],
    # png
    [(5_000, 5_000), "png", 4],
    [(10_000, 10_000), "png", 9],
    [(10_000, 10_000, 5), "png", 36],
    [(10_000, 5_000), "png", 6],
    [(100_000_000,), "png", 7],
])
def test_simple(config):
    # for simple cases, the shape optimization should find the global minimum energy state
    sample_shape, sample_compression, expected_num_tiles = config

    tensor_meta = TensorMeta("generic", sample_compression=sample_compression, dtype="int32")
    tensor_meta.max_chunk_size = 32 * MB
    tensor_meta.min_chunk_size = 16 * MB

    actual_tile_shape = optimize_tile_shape(sample_shape, tensor_meta)
    actual_num_tiles = num_tiles_for_sample(actual_tile_shape, sample_shape)
    actual_num_bytes_per_tile = approximate_num_bytes(actual_tile_shape, tensor_meta)

    msg = f"tile_shape={actual_tile_shape}, num_tiles={actual_num_tiles}, num_bytes={actual_num_bytes_per_tile}"

    assert expected_num_tiles == actual_num_tiles, msg
    assert actual_num_bytes_per_tile >= tensor_meta.min_chunk_size, msg
    assert actual_num_bytes_per_tile <= tensor_meta.max_chunk_size, msg


@pytest.mark.parametrize("sample_shape", [
    (1, 1, 1, 10_000, 100_000_000, 5, 33, 99999999999),
    (999, 333, 666666, 1, 5555, 1000_0000_0000, 5),
    (9_000_000_000_000, 5555555555555),
])
@pytest.mark.parametrize("sample_compression", [None, "png"])
def test_complex(sample_shape, sample_compression):
    # for complex cases, the shape optimization should find an energy state good enough

    tensor_meta = TensorMeta("generic", sample_compression=sample_compression, dtype="int32")
    tensor_meta.max_chunk_size = 32 * MB
    tensor_meta.min_chunk_size = 16 * MB

    actual_tile_shape = optimize_tile_shape(sample_shape, tensor_meta)
    actual_num_bytes_per_tile = approximate_num_bytes(actual_tile_shape, tensor_meta)

    assert actual_num_bytes_per_tile <= tensor_meta.max_chunk_size
    assert actual_num_bytes_per_tile >= tensor_meta.min_chunk_size