from typing import Optional
from hub.util.exceptions import CannotInferTilesError
import pytest
import numpy as np
from hub.constants import KB


def _assert_num_chunks(
    actual_num_chunks: int, expected_num_chunks: int, compression: Optional[str]
):
    if compression is None:
        assert actual_num_chunks == expected_num_chunks
    else:
        # TODO: better way to get the number of chunks with compression for tests
        assert actual_num_chunks < expected_num_chunks


@pytest.mark.parametrize("compression", [None, "png"])
def test_initialize_large_tensor(local_ds_generator, compression):
    ds = local_ds_generator()

    # keep max chunk size default, this test should run really fast since we barely fill in any data
    ds.create_tensor("tensor", dtype="int32", sample_compression=compression)

    ds.tensor.append_empty((10000, 10000))  # 400MB
    _assert_num_chunks(ds.tensor.num_chunks, 16, compression)

    ds = local_ds_generator()
    assert ds.tensor.shape == (1, 10000, 10000)
    np.testing.assert_array_equal(
        ds.tensor[0, 5500:5510, 5500:5510].numpy(), np.zeros((10, 10), dtype="int32")
    )

    # fill in some data
    ds.tensor[0, 9050:9055, 9050:9055] = np.ones((5, 5), dtype="int32")

    ds = local_ds_generator()
    actual = ds.tensor[0, 9050:9060, 9050:9060].numpy()
    expected = np.zeros((10, 10), dtype="int32")
    expected[0:5, 0:5] = 1
    np.testing.assert_array_equal(actual, expected)

    assert ds.tensor.shape == (1, 10000, 10000)

    _assert_num_chunks(ds.tensor.num_chunks, 16, compression)


@pytest.mark.parametrize("compression", [None, "png"])
def test_initialize_large_image(local_ds_generator, compression):
    ds = local_ds_generator()

    # keep max chunk size default, this test should run really fast since we barely fill in any data
    ds.create_tensor("tensor", dtype="uint8", sample_compression=compression)

    ds.tensor.append_empty((10, 10, 3))  # small
    ds.tensor.append_empty((10000, 10000, 3))  # large (1.2GB) w/ channel dim
    ds.tensor.append(np.ones((10, 10, 3), dtype="uint8"))  # small
    ds.tensor.extend_empty((5, 10, 10, 3))  # small

    _assert_num_chunks(ds.tensor.num_chunks, 18, compression)

    ds = local_ds_generator()
    assert ds.tensor.shape == (8, None, None, 3)
    np.testing.assert_array_equal(ds.tensor[0].numpy(), np.zeros((10, 10, 3)))
    np.testing.assert_array_equal(
        ds.tensor[1, 50:100, 50:100, :].numpy(), np.zeros((50, 50, 3))
    )
    np.testing.assert_array_equal(
        ds.tensor[1, -100:-50, -100:-50, :].numpy(), np.zeros((50, 50, 3))
    )
    np.testing.assert_array_equal(
        ds.tensor[1, -100:-50, -100:-50, :].numpy(), np.zeros((50, 50, 3))
    )

    # update large sample (only filling in 10KB of data)
    ds = local_ds_generator()
    ds.tensor[1, 50:100, 50:100, 0] = np.ones((1, 50, 50), dtype="uint8")
    ds.tensor[1, 50:100, 50:100, 1] = np.ones((1, 50, 50), dtype="uint8") * 2
    ds.tensor[1, 50:100, 50:100, 2] = np.ones((1, 50, 50), dtype="uint8") * 3

    ds = local_ds_generator()
    expected = np.ones((50, 50, 3), dtype="uint8")
    expected[:, :, 1] *= 2
    expected[:, :, 2] *= 3
    np.testing.assert_array_equal(ds.tensor[1, 50:100, 50:100, :].numpy(), expected)

    _assert_num_chunks(ds.tensor.num_chunks, 18, compression)


@pytest.mark.parametrize("compression", [None, "png"])
def test_populate_full_large_sample(local_ds_generator, compression):
    ds = local_ds_generator()

    ds.create_tensor(
        "large",
        dtype="int32",
        sample_compression=compression,
        max_chunk_size=16 * KB,
    )

    ds.large.append_empty((500, 500))  # 1MB, ~63 chunks (uncompressed)

    assert ds.large.shape == (1, 500, 500)
    _assert_num_chunks(ds.large.num_chunks, 64, compression)

    # if patch size is equal to the tile shape on all dimensions, then no cross-tile updates are made
    patch_size = 50

    with ds:
        patch_count = 0
        last_x = 0
        last_y = 0
        for x in range(patch_size, 500 + patch_size, patch_size):
            for y in range(patch_size, 500 + patch_size, patch_size):
                print(last_x, x, last_y, y)
                patch = np.ones((patch_size, patch_size), dtype="int32") * patch_count
                ds.large[0, last_x:x, last_y:y] = patch
                last_y = y
            last_x = x
            last_y = 0

    ds = local_ds_generator()
    _assert_num_chunks(ds.large.num_chunks, 64, compression)

    # check data
    patch_count = 0
    last_x = 0
    last_y = 0
    for x in range(patch_size, 500 + patch_size, patch_size):
        for y in range(patch_size, 500 + patch_size, patch_size):
            print(last_x, x, last_y, y)
            expected_patch = np.ones((patch_size, patch_size), dtype="int32") * patch_count
            actual_patch = ds.large[0, last_x:x, last_y:y].numpy()
            np.testing.assert_array_equal(expected_patch, actual_patch, f"x={last_x}:{x}, y={last_y}:{y}")
            last_y = y
        last_x = x
        last_y = 0

    assert ds.large.shape == (1, 500, 500)

    # TODO: uncomment after replacing tiled samples is implemented
    # ds = local_ds_generator()
    # np.testing.assert_array_equal(ds.large.numpy(), np.ones((5, 5), dtype="int32") * 4)
    # assert ds.large.shape == (1, 5, 5)
    # assert ds.large.num_chunks == 1


def test_failures(memory_ds):
    memory_ds.create_tensor("tensor")

    _assert_num_chunks(memory_ds.tensor.num_chunks, 0, None)
    with pytest.raises(CannotInferTilesError):
        # dtype must be pre-defined before an empty sample can be created (otherwise we can't infer the num chunks)
        memory_ds.tensor.append_empty((10000, 10000))
    assert memory_ds.tensor.shape == (0,)
    _assert_num_chunks(memory_ds.tensor.num_chunks, 0, None)

    # fix
    memory_ds.tensor.set_dtype("uint8")
    memory_ds.tensor.append_empty((10000, 10000))
    assert memory_ds.tensor.shape == (1, 10000, 10000)
    assert memory_ds.tensor[0, 0:5, 0:5].numpy().dtype == np.dtype("uint8")

    # TODO: replace tiled sample with a non-tiled sample
    with pytest.raises(NotImplementedError):
        memory_ds.tensor[0] = np.ones((5, 5), dtype="int32") * 4
