import hub
from hub.util.exceptions import CannotInferTilesError
import pytest
import numpy as np
from hub.constants import B, KB
from hub.tests.common import assert_array_lists_equal, compressions



def _get_random_image(shape):
    return np.random.randint(low=0, high=256, size=np.prod(shape), dtype="uint8").reshape(shape)


def _assert_num_chunks(
    actual_num_chunks: int, expected_num_chunks: int, compression: dict
):
    # TODO: uncomment when tile optimization is more consistent
    # if compression is None:
    #     is_compressed = False
    # else:
    #     is_compressed = list(compression.values())[0] is not None
    #     assert len(compression.values()) == 1

    # if is_compressed:
    #     assert actual_num_chunks <= expected_num_chunks
    # else:
    #     assert actual_num_chunks == expected_num_chunks

    assert actual_num_chunks <= expected_num_chunks


@compressions
def test_initialize_large_tensor(local_ds_generator, compression):
    ds = local_ds_generator()

    # keep max chunk size default, this test should run really fast since we barely fill in any data
    ds.create_tensor("tensor", dtype="uint8", **compression)

    ds.tensor.append_empty((10000, 10000))  # 100MB uncompressed
    _assert_num_chunks(ds.tensor.num_chunks, 4, compression)

    ds = local_ds_generator()
    assert ds.tensor.shape == (1, 10000, 10000)
    np.testing.assert_array_equal(
        ds.tensor[0, 5500:5510, 5500:5510].numpy(), np.zeros((10, 10), dtype="uint8")
    )

    # fill in some data
    ds.tensor[0, 9050:9055, 9050:9055] = np.ones((5, 5), dtype="uint8")

    ds = local_ds_generator()
    actual = ds.tensor[0, 9050:9060, 9050:9060].numpy()
    expected = np.zeros((10, 10), dtype="uint8")
    expected[0:5, 0:5] = 1
    np.testing.assert_array_equal(actual, expected)

    assert ds.tensor.shape == (1, 10000, 10000)

    _assert_num_chunks(ds.tensor.num_chunks, 4, compression)


@compressions
def test_num_chunks(local_ds_generator, compression):
    ds = local_ds_generator()

    # keep max chunk size default, this test should run really fast since we barely fill in any data
    ds.create_tensor("tensor", dtype="uint8", **compression)

    ds.tensor.append_empty((10, 10, 3))  # small
    _assert_num_chunks(ds.tensor.num_chunks, 1, compression)

    ds.tensor.append_empty((10000, 10000, 3))  # large (300MB) w/ channel dim
    _assert_num_chunks(ds.tensor.num_chunks, 17, compression)

    ds.tensor.append(np.ones((10, 10, 3), dtype="uint8"))  # small
    _assert_num_chunks(ds.tensor.num_chunks, 18, compression)
    ds.tensor.extend_empty((5, 10, 10, 3))  # small
    _assert_num_chunks(ds.tensor.num_chunks, 18, compression)


@compressions
def test_populate_full_large_sample(local_ds_generator, compression):
    ds = local_ds_generator()

    ds.create_tensor(
        "large",
        dtype="int32",
        **compression,
        max_chunk_size=200 * KB,
    )

    ds.large.append_empty((500, 500))  # 1MB

    assert ds.large.shape == (1, 500, 500)
    _assert_num_chunks(ds.large.num_chunks, 9, compression)

    # if patch size is equal to the tile shape on all dimensions, then no cross-tile updates are made
    patch_size = 50

    with ds:
        patch_count = 0
        last_x = 0
        last_y = 0
        for x in range(patch_size, 500 + patch_size, patch_size):
            for y in range(patch_size, 500 + patch_size, patch_size):
                patch = np.ones((patch_size, patch_size), dtype="int32") * patch_count
                ds.large[0, last_x:x, last_y:y] = patch
                last_y = y
            last_x = x
            last_y = 0

    ds = local_ds_generator()
    _assert_num_chunks(ds.large.num_chunks, 9, compression)

    # check data
    patch_count = 0
    last_x = 0
    last_y = 0
    for x in range(patch_size, 500 + patch_size, patch_size):
        for y in range(patch_size, 500 + patch_size, patch_size):
            expected_patch = np.ones((patch_size, patch_size), dtype="int32") * patch_count
            actual_patch = ds.large[0, last_x:x, last_y:y].numpy()
            np.testing.assert_array_equal(expected_patch, actual_patch, f"x={last_x}:{x}, y={last_y}:{y}")
            last_y = y
        last_x = x
        last_y = 0

    assert ds.large.shape == (1, 500, 500)


def test_failures(memory_ds):
    memory_ds.create_tensor("tensor")

    _assert_num_chunks(memory_ds.tensor.num_chunks, 0, None)
    with pytest.raises(CannotInferTilesError):
        # dtype must be pre-defined before an empty sample can be created (otherwise we can't infer the num chunks)
        memory_ds.tensor.append_empty((10000, 10000))
    assert memory_ds.tensor.shape == (0,)
    _assert_num_chunks(memory_ds.tensor.num_chunks, 0, None)

    # fix for failure
    memory_ds.tensor.set_dtype("uint8")
    memory_ds.tensor.append_empty((10000, 10000))
    assert memory_ds.tensor.shape == (1, 10000, 10000)
    assert memory_ds.tensor[0, 0:5, 0:5].numpy().dtype == np.dtype("uint8")

    # TODO: implement re-tiling
    with pytest.raises(NotImplementedError):
        memory_ds.tensor[0] = np.ones((5, 5), dtype="int32") * 4


@compressions
def test_read_accross_boundaries(memory_ds, compression):
    tensor = memory_ds.create_tensor("tensor", dtype="uint8", **compression, max_chunk_size=1 * KB)
    
    x = _get_random_image((150, 150))
    tensor.append_empty((150, 150))
    tensor[0, 0:150, 0:150] = x.copy()
    assert tensor[0].num_chunks == 25

    assert_array_lists_equal(tensor[0, 10:12, 1:5].numpy(), x[10:12, 1:5])
    assert_array_lists_equal(tensor[0, 10:50, 1:50].numpy(), x[10:50, 1:50])

    # read accross multiple tile boundaries
    assert_array_lists_equal(tensor[0, 10:130, 1:100].numpy(), x[10:130, 1:100])


@compressions
def test_trivial_indexing(memory_ds, compression):
    memory_ds.create_tensor("tensor", dtype="uint8", max_chunk_size=1 * KB, **compression)
    _assert_num_chunks(memory_ds.tensor.num_chunks, 0, None)

    memory_ds.tensor.append_empty((100, 100))
    _assert_num_chunks(memory_ds.tensor.num_chunks, 16, None)

    x = np.arange(100*100, dtype="uint8").reshape((100, 100))

    memory_ds.tensor[0, 0:100, 0:100] = x.copy()
    np.testing.assert_array_equal(memory_ds.tensor[0].numpy(), x)

    y = x + 5

    memory_ds.tensor.append(y.copy())
    _assert_num_chunks(memory_ds.tensor.num_chunks, 32, None)
    expected = [x, y]
    assert_array_lists_equal(memory_ds.tensor.numpy(), expected)


@compressions
def test_append(local_ds, compression, tatevik):
    large1 = _get_random_image((90, 100, 4))
    large2 = _get_random_image((100, 90, 4))
    small = _get_random_image((10, 10, 4))

    local_ds.create_tensor("image", dtype="uint8", **compression, max_chunk_size=20 * KB)

    local_ds.image.append(large1.copy())
    _assert_num_chunks(local_ds.image.num_chunks, 4, compression)
    local_ds.image.append(small.copy())
    _assert_num_chunks(local_ds.image.num_chunks, 5, compression)
    local_ds.image.append(large2.copy())
    _assert_num_chunks(local_ds.image.num_chunks, 9, compression)
    local_ds.image.append(hub.read(tatevik))
    _assert_num_chunks(local_ds.image.num_chunks, 73, compression)

    assert local_ds.image.shape_interval.lower == (4, 10, 10, 4)
    assert local_ds.image.shape_interval.upper == (4, 496, 498, 4)

    np.testing.assert_array_equal(small, local_ds.image[1].numpy())
    np.testing.assert_array_equal(large1, local_ds.image[0, 0:90, 0:100, 0:4].numpy())

    expected = [large1, small, large2, hub.read(tatevik).array]
    assert_array_lists_equal(expected, local_ds.image.numpy(aslist=True))


@compressions
def test_extend(local_ds, compression, davit):
    local_ds.create_tensor("image", dtype="uint8", **compression, max_chunk_size=10 * KB)

    small1 = _get_random_image((10, 10, 3))
    small2 = _get_random_image((5, 20, 3))
    large = _get_random_image((100, 100, 3))

    local_ds.image.extend([
        small1.copy(),
        small2.copy(),
        hub.read(davit),
        small2.copy(),
        large.copy(),
        small2.copy(),
        small1.copy(),
    ])
    _assert_num_chunks(local_ds.image.num_chunks, 23, compression)

    assert local_ds.image.shape_interval.lower == (7, 5, 10, 3)
    assert local_ds.image.shape_interval.upper == (7, 200, 200, 3)

    expected = [small1, small2, hub.read(davit).array, small2, large, small2, small1]

    assert local_ds.image[4, 100:120, 100:120, :].numpy().size == 0
    assert local_ds.image[4, 90:120, 90:120, :].numpy().shape == (10, 10, 3)

    assert_array_lists_equal(expected, local_ds.image.numpy(aslist=True))