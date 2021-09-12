from hub.core.compression import get_compression_factor
import hub
from re import A
import numpy as np
from hub.constants import DEFAULT_MAX_CHUNK_SIZE, KB, MB, UNSPECIFIED
from hub.util.tiles import approximate_num_bytes, num_tiles_for_sample
from hub.core.meta.tensor_meta import TensorMeta
from hub.core.tiling.optimize import TileOptimizer
import pytest


# consistent simulated annealing
np.random.seed(1)


# for debugging, you can use this visualization script:
# https://gist.github.com/mccrearyd/d4a9506813fe64d42c63b090997d9145


tile_test_compressions = pytest.mark.parametrize("compression", [None, "png"])


def _get_tensor_meta(sample_compression: str) -> TensorMeta:
    # TODO: test with chunk-wise compression
    tensor_meta = TensorMeta(
        "generic", sample_compression=sample_compression, chunk_compression=UNSPECIFIED, dtype="int32"
    )
    return tensor_meta


def _get_optimizer(sample_compression: str) -> TileOptimizer:
    tensor_meta = _get_tensor_meta(sample_compression)
    optimizer = TileOptimizer(DEFAULT_MAX_CHUNK_SIZE // 2, DEFAULT_MAX_CHUNK_SIZE, tensor_meta)
    return optimizer


def _assert_valid(optimizer, sample_shape, expected_num_tiles=None, compression_factor=None):
    tensor_meta = optimizer.tensor_meta

    tile_shape = optimizer.last_tile_shape
    actual_num_tiles = num_tiles_for_sample(tile_shape, sample_shape)

    compression_factor = compression_factor or get_compression_factor(tensor_meta)
    nbytes_per_tile = approximate_num_bytes(tile_shape, tensor_meta.dtype, compression_factor)

    msg = f"tile_shape={tile_shape}, num_tiles={actual_num_tiles}, num_bytes_per_tile={nbytes_per_tile}, sample_shape={sample_shape}"

    if expected_num_tiles is not None:
        assert actual_num_tiles <= expected_num_tiles, msg
    assert nbytes_per_tile <= optimizer.max_chunk_size, msg
    assert nbytes_per_tile >= optimizer.min_chunk_size, msg


# sample_shape, sample_compression, expected_num_tiles
@pytest.mark.parametrize(
    "config",
    [
        # no compression
        [(5_000, 5_000), None, 4],
        [(10_000, 10_000), None, 16],
        [(10_000, 10_000, 5), None, 64],
        [(10_000, 5_000), None, 8],
        [(100_000_000,), None, 13],
        # png
        [(5_000, 5_000), "png", 4],
        [(10_000, 10_000), "png", 9],
        [(10_000, 10_000, 5), "png", 30],
        [(10_000, 5_000), "png", 6],
        [(100_000_000,), "png", 5],
    ],
)
def test_simple(config):
    # for simple cases, the shape optimization should find the global minimum energy state
    sample_shape, sample_compression, expected_num_tiles = config

    optimizer = _get_optimizer(sample_compression)
    optimizer.optimize(sample_shape)

    _assert_valid(optimizer, sample_shape, expected_num_tiles=expected_num_tiles)


@pytest.mark.parametrize(
    "sample_shape",
    [
        (1, 1, 1, 10_000, 100_000_000, 5, 33, 99999999999),
        (999, 333, 666666, 1, 5555, 1000_0000_0000, 5),
        (9_000_000_000_000, 5555555555555),
    ],
)
@tile_test_compressions
def test_complex(sample_shape, compression):
    # for complex cases, the shape optimization should find an energy state good enough
    
    optimizer = _get_optimizer(compression)

    optimizer.optimize(sample_shape)
    _assert_valid(optimizer, sample_shape)


# sample_shape, sample_compression, compression_factor, expected_num_tiles
@pytest.mark.parametrize(
    "config",
    [
        [(90, 100, 3), None, 1, 9],
    ]
)
def test_explicit_factor(config):
    sample_shape, compression, compression_factor, expected_num_tiles = config

    optimizer = _get_optimizer(compression)

    optimizer.min_chunk_size = 10 * KB
    optimizer.max_chunk_size = 20 * KB
    optimizer.optimize(sample_shape, compression_factor=compression_factor, validate=True)

    # no matter the compression, it should always be consistent since compression factor is determined
    _assert_valid(optimizer, sample_shape, expected_num_tiles=expected_num_tiles, compression_factor=compression_factor)