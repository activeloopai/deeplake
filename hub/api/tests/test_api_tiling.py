import pytest
import numpy as np

compressions_paremetrized = pytest.mark.parametrize(
    "compression",
    [
        {"sample_compression": None},
        {"sample_compression": "png"},
        {"chunk_compression": "png"},
        {"sample_compression": "lz4"},
        {"chunk_compression": "lz4"},
    ],
)


def test_simple(memory_ds):
    with memory_ds:
        memory_ds.create_tensor("abc")
        memory_ds.abc.extend(np.ones((3, 1003, 2001, 5)))
    np.testing.assert_array_equal(memory_ds.abc.numpy(), np.ones((3, 1003, 2001, 5)))


@compressions_paremetrized
def test_mixed_small_large(local_ds_generator, compression):
    ds = local_ds_generator()
    arr1 = np.random.randint(0, 255, (3003, 2001, 3)).astype(np.uint8)
    arr2 = np.random.randint(0, 255, (500, 500, 3)).astype(np.uint8)
    arr3 = np.random.randint(0, 255, (2503, 2501, 3)).astype(np.uint8)
    with ds:
        ds.create_tensor("abc", max_chunk_size=2 ** 21.0 ** compression)
        for i in range(10):
            if i % 5 == 0:
                ds.abc.append(arr1)
            else:
                ds.abc.append(arr2)

    for i in range(10):
        if i % 5 == 0:
            np.testing.assert_array_equal(ds.abc[i].numpy(), arr1)
        else:
            np.testing.assert_array_equal(ds.abc[i].numpy(), arr2)

    ds = local_ds_generator()
    for i in range(10):
        if i % 5 == 0:
            np.testing.assert_array_equal(ds.abc[i].numpy(), arr1)
        else:
            np.testing.assert_array_equal(ds.abc[i].numpy(), arr2)

    with ds:
        ds.abc.extend([arr3] * 3)

    for i in range(13):
        if i >= 10:
            np.testing.assert_array_equal(ds.abc[i].numpy(), arr3)
        elif i % 5 == 0:
            np.testing.assert_array_equal(ds.abc[i].numpy(), arr1)
        else:
            np.testing.assert_array_equal(ds.abc[i].numpy(), arr2)

    ds = local_ds_generator()
    for i in range(13):
        if i >= 10:
            np.testing.assert_array_equal(ds.abc[i].numpy(), arr3)
        elif i % 5 == 0:
            np.testing.assert_array_equal(ds.abc[i].numpy(), arr1)
        else:
            np.testing.assert_array_equal(ds.abc[i].numpy(), arr2)


@compressions_paremetrized
def test_updates(memory_ds, compression):
    arr1 = np.random.randint(0, 255, (3003, 2001, 3)).astype(np.uint8)
    arr2 = np.random.randint(0, 255, (500, 500, 3)).astype(np.uint8)
    arr3 = np.random.randint(0, 255, (2503, 2501, 3)).astype(np.uint8)
    arr4 = np.random.randint(0, 255, (250, 250, 3)).astype(np.uint8)
    with memory_ds:
        memory_ds.create_tensor("abc", max_chunk_size=2 ** 21, **compression)
        for i in range(10):
            if i % 5 == 0:
                memory_ds.abc.append(arr1)
            else:
                memory_ds.abc.append(arr2)

    with memory_ds:
        for i in range(10):
            if i % 5 != 0:
                memory_ds.abc[i] = arr3 if i % 2 == 0 else arr4

    for i in range(10):
        if i % 5 == 0:
            np.testing.assert_array_equal(memory_ds.abc[i].numpy(), arr1)
        elif i % 2 == 0:
            np.testing.assert_array_equal(memory_ds.abc[i].numpy(), arr3)
        else:
            np.testing.assert_array_equal(memory_ds.abc[i].numpy(), arr4)
