import pytest

import numpy as np

import deeplake
from deeplake.core.compression import decompress_array
from deeplake.core.serialize import (
    serialize_numpy_and_base_types,
    serialize_sample_object,
)
from deeplake.constants import KB, MB
from deeplake.core.tiling.deserialize import np_list_to_sample
from deeplake.core.tiling.sample_tiles import SampleTiles


def test_numpy_and_base_types():
    arr1 = np.ones((101, 100, 3), dtype=np.int32)
    out1, shape = serialize_numpy_and_base_types(
        arr1, None, None, "int64", "generic", 16 * MB
    )
    arr1_deserialized = np.frombuffer(out1, dtype=np.int64).reshape(shape)
    np.testing.assert_array_equal(arr1, arr1_deserialized)

    out2, shape = serialize_numpy_and_base_types(
        arr1, None, None, "int64", "generic", 100 * KB
    )
    assert isinstance(out2, SampleTiles)
    out_list = [out2.yield_tile() for _ in range(out2.num_tiles)]
    np_list = [np.frombuffer(b[0], dtype=np.int64).reshape(b[1]) for b in out_list]
    tile_shape, layout_shape = out2.tile_shape, out2.layout_shape
    out2 = np_list_to_sample(np_list, shape, tile_shape, layout_shape, "int64")
    np.testing.assert_array_equal(arr1, out2)

    arr2 = np.random.randint(0, 255, (501, 503, 3), dtype="uint8")
    out2, shape = serialize_numpy_and_base_types(
        arr2, "png", None, "uint8", "generic", 16 * MB
    )
    arr2_deserialized = decompress_array(out2, compression="png").reshape(shape)
    np.testing.assert_array_equal(arr2, arr2_deserialized)

    out3, shape = serialize_numpy_and_base_types(
        arr2, "png", None, "uint8", "generic", 100 * KB
    )
    assert isinstance(out3, SampleTiles)
    out_list = [out3.yield_tile() for _ in range(out3.num_tiles)]
    np_list = [
        decompress_array(b[0], compression="png").reshape(b[1]) for b in out_list
    ]
    tile_shape, layout_shape = out3.tile_shape, out3.layout_shape
    out3 = np_list_to_sample(np_list, shape, tile_shape, layout_shape, "uint8")
    np.testing.assert_array_equal(arr2, out3)


@pytest.mark.slow
def test_sample_img_compression(cat_path, compression="png"):
    sample = deeplake.read(cat_path)
    arr = sample.array

    # reloaded to get rid of cached array in sample
    sample = deeplake.read(cat_path)
    out, shape = serialize_sample_object(
        sample, compression, None, "uint16", "generic", 16 * MB
    )
    arr_deserialized = decompress_array(out, compression=compression).reshape(shape)
    np.testing.assert_array_equal(arr, arr_deserialized)

    # reloaded to get rid of cached array in sample
    sample = deeplake.read(cat_path)
    out, shape = serialize_sample_object(
        sample, compression, None, "uint16", "generic", 100 * KB
    )
    assert isinstance(out, SampleTiles)
    out_list = [out.yield_tile() for _ in range(out.num_tiles)]
    np_list = [
        decompress_array(b[0], compression=compression).reshape(b[1]) for b in out_list
    ]
    tile_shape, layout_shape = out.tile_shape, out.layout_shape
    out = np_list_to_sample(np_list, shape, tile_shape, layout_shape, "uint16")
    np.testing.assert_array_equal(arr, out)


def test_sample_byte_compression(cat_path, compression="lz4"):
    sample = deeplake.read(cat_path)
    arr = sample.array

    # reloaded to get rid of cached array in sample
    sample = deeplake.read(cat_path)
    dtype = "uint16"
    out, shape = serialize_sample_object(
        sample, compression, None, dtype, "generic", 16 * MB
    )
    arr_deserialized = decompress_array(out, shape, dtype, compression).reshape(shape)
    np.testing.assert_array_equal(arr, arr_deserialized)

    # reloaded to get rid of cached array in sample
    sample = deeplake.read(cat_path)
    out, shape = serialize_sample_object(
        sample, compression, None, dtype, "generic", 100 * KB
    )
    assert isinstance(out, SampleTiles)
    out_list = [out.yield_tile() for _ in range(out.num_tiles)]
    np_list = [decompress_array(b[0], b[1], dtype, compression) for b in out_list]
    tile_shape, layout_shape = out.tile_shape, out.layout_shape
    out = np_list_to_sample(np_list, shape, tile_shape, layout_shape, "uint16")
    np.testing.assert_array_equal(arr, out)


def test_sample_no_compression(cat_path):
    sample = deeplake.read(cat_path)
    arr = sample.array

    # reloaded to get rid of cached array in sample
    sample = deeplake.read(cat_path)
    out, shape = serialize_sample_object(
        sample, None, None, "uint16", "generic", 16 * MB
    )
    arr_deserialized = np.frombuffer(out, dtype="uint16").reshape(shape)
    np.testing.assert_array_equal(arr, arr_deserialized)

    # reloaded to get rid of cached array in sample
    sample = deeplake.read(cat_path)
    out, shape = serialize_sample_object(
        sample, None, None, "uint16", "generic", 100 * KB
    )
    assert isinstance(out, SampleTiles)
    out_list = [out.yield_tile() for _ in range(out.num_tiles)]
    np_list = [np.frombuffer(b[0], dtype="uint16").reshape(b[1]) for b in out_list]
    tile_shape, layout_shape = out.tile_shape, out.layout_shape
    out = np_list_to_sample(np_list, shape, tile_shape, layout_shape, "uint16")
    np.testing.assert_array_equal(arr, out)
