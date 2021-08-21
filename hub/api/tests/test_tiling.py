from hub.util.exceptions import CannotInferTilesError
import pytest
import numpy as np
from hub.constants import KB


@pytest.mark.parametrize("compression", [None, "png"])
def test_initialize_large_tensor(local_ds_generator, compression):
    ds = local_ds_generator()

    # keep max chunk size default, this test should run really fast since we barely fill in any data
    ds.create_tensor("tensor", dtype="int32", sample_compression=compression)

    ds.tensor.append_empty((10000, 10000))  # 400MB

    ds = local_ds_generator()
    assert ds.tensor.shape == (1, 10000, 10000)
    np.testing.assert_array_equal(ds.tensor[0:10, 0:10].numpy(), np.zeros((10, 10), dtype="int32"))

    # fill in some data
    ds.tensor[0:5, 0:5] = np.ones((5, 5), dtype="int32")

    ds = local_ds_generator()
    actual = ds.tensor[0:10, 0:10].numpy()
    expected = np.zeros((10, 10), dtype="int32")
    expected[0:5, 0:5] = 1
    np.testing.assert_array_equal(actual, expected) 

    assert ds.tensor.shape == (1, 10000, 10000)


@pytest.mark.parametrize("compression", [None, "png"])
def test_initialize_large_image(local_ds_generator, compression):
    ds = local_ds_generator()

    # keep max chunk size default, this test should run really fast since we barely fill in any data
    ds.create_tensor("tensor", dtype="int32", sample_compression=compression)

    ds.tensor.append_empty((10, 10, 3))  # small
    ds.tensor.append_empty((10000, 10000, 3))  # large (1.2GB) w/ channel dim
    ds.tensor.append(np.ones((10, 10, 3), dtype="int32"))  # small
    ds.tensor.extend_empty((5, 10, 10, 3))  # small

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
    ds.tensor[1, 50:100, 50:100, 0] = np.ones((1, 50, 50, 1), dtype="int32")
    ds.tensor[1, 50:100, 50:100, 1] = np.ones((1, 50, 50, 1), dtype="int32") * 2
    ds.tensor[1, 50:100, 50:100, 2] = np.ones((1, 50, 50, 1), dtype="int32") * 3

    ds = local_ds_generator()
    expected = np.ones((50, 50, 3), dtype="int32")
    expected[:, :, 1] *= 2
    expected[:, :, 2] *= 3
    np.testing.assert_array_equal(ds.tensor[1, 50:100, 50:100, :].numpy(), expected)


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

    # update large sample in 50x50 patches
    with ds:
        patch_count = 0
        last_x = 0
        last_y = 0
        for x in range(0, 500, 50):
            for y in range(0, 500, 50):
                patch = np.ones((50, 50), dtype="int32") * patch_count
                ds.large[last_x:x, last_y:y] = patch
                last_y = y
            last_x = x

    ds = local_ds_generator()

    if compression is None:
        assert ds.large.num_chunks == 64
    else:
        assert ds.large.num_chunks < 63

    # check data
    patch_count = 0
    last_x = 0
    last_y = 0
    for x in range(0, 500, 50):
        for y in range(0, 500, 50):
            expected_patch = np.ones((50, 50), dtype="int32") * patch_count
            actual_patch = ds.large[last_x:x, last_y:y].numpy()
            np.testing.assert_array_equal(expected_patch, actual_patch)
            last_y = y
        last_x = x

    assert ds.large.shape == (1, 500, 500)


def test_failures(memory_ds):
    memory_ds.create_tensor("tensor")

    with pytest.raises(CannotInferTilesError):
        # dtype must be pre-defined before an empty sample can be created (otherwise we can't infer the num chunks)
        memory_ds.tensor.append_empty((10000, 10000))
    assert memory_ds.tensor.shape == (0,)

    # fix
    memory_ds.tensor.set_dtype("uint8")
    memory_ds.tensor.append_empty((10000, 10000))
    assert memory_ds.tensor.shape == (1, 10000, 10000)
    assert memory_ds.tensor[0:5, 0:5].numpy().dtype == np.dtype("uint8")
