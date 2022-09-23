import pytest
import numpy as np

from deeplake.constants import MB

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
        memory_ds.create_tensor("abc", max_chunk_size=2 * MB, tiling_threshold=2 * MB)
        memory_ds.abc.extend(np.ones((3, 253, 501, 5)))
    np.testing.assert_array_equal(memory_ds.abc.numpy(), np.ones((3, 253, 501, 5)))
    memory_ds.commit()
    np.testing.assert_array_equal(memory_ds.abc.numpy(), np.ones((3, 253, 501, 5)))


@compressions_paremetrized
def test_mixed_small_large(local_ds_generator, compression):
    ds = local_ds_generator()
    arr1 = np.random.randint(0, 255, (3003, 2001, 3)).astype(np.uint8)
    arr2 = np.random.randint(0, 255, (500, 500, 3)).astype(np.uint8)
    arr3 = np.random.randint(0, 255, (250, 250, 3)).astype(np.uint8)

    idxs = [
        (slice(73, 117), slice(9, 17)),
        4,
        -1,
        slice(
            10,
        ),
        slice(20, 37),
    ]

    with ds:
        ds.create_tensor(
            "abc", max_chunk_size=2**21, tiling_threshold=2**20, **compression
        )
        for i in range(10):
            if i % 5 == 0:
                ds.abc.append(arr1)
            else:
                ds.abc.append(arr2)

    for i in range(10):
        if i % 5 == 0:
            np.testing.assert_array_equal(ds.abc[i].numpy(), arr1)
            for idx in idxs:
                np.testing.assert_array_equal(ds.abc[i][idx].numpy(), arr1[idx])
        else:
            np.testing.assert_array_equal(ds.abc[i].numpy(), arr2)
            for idx in idxs:
                np.testing.assert_array_equal(ds.abc[i][idx].numpy(), arr2[idx])

    ds = local_ds_generator()

    for i in range(10):
        if i % 5 == 0:
            np.testing.assert_array_equal(ds.abc[i].numpy(), arr1)
            for idx in idxs:
                np.testing.assert_array_equal(ds.abc[i][idx].numpy(), arr1[idx])
        else:
            np.testing.assert_array_equal(ds.abc[i].numpy(), arr2)
            for idx in idxs:
                np.testing.assert_array_equal(ds.abc[i][idx].numpy(), arr2[idx])

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

    update_idx = (slice(73, 117), slice(1765, 1901))

    arr5 = arr1 * 2
    arr6 = arr5[update_idx]
    arr6 += 1

    with memory_ds:
        memory_ds.create_tensor(
            "abc", max_chunk_size=2**21, tiling_threshold=2**20, **compression
        )
        for i in range(10):
            if i % 5 == 0:
                memory_ds.abc.append(arr1)
            else:
                memory_ds.abc.append(arr2)
            len(memory_ds)
    with memory_ds:
        for i in range(10):
            if i % 5 == 0:
                memory_ds.abc[i] = arr1 * 2
                memory_ds.abc[i][update_idx] = arr6
            else:
                memory_ds.abc[i] = arr3 if i % 2 == 0 else arr4

    for i in range(10):
        if i % 5 == 0:
            np.testing.assert_array_equal(memory_ds.abc[i].numpy(), arr5)
        elif i % 2 == 0:
            np.testing.assert_array_equal(memory_ds.abc[i].numpy(), arr3)
        else:
            np.testing.assert_array_equal(memory_ds.abc[i].numpy(), arr4)

    # update tiled sample with small sample
    arr7 = np.random.randint(0, 255, (3, 2, 3)).astype(np.uint8)
    memory_ds.abc[0] = arr7
    np.testing.assert_array_equal(memory_ds.abc[0].numpy(), arr7)


def test_cachable_overflow(memory_ds):
    ds = memory_ds
    with ds:
        ds.create_tensor("x")
        ds.create_tensor("y")
        ds.x.extend(np.ones((3, 4000, 3000)))
        ds.y.extend(np.ones((3, 4000, 3000)))
    assert len(ds) == 3
    assert len(ds.x) == 3
    assert len(ds.y) == 3


@compressions_paremetrized
def test_empty_array(memory_ds, compression):
    ds = memory_ds
    arr_list = [
        np.random.randint(0, 255, (3894, 0, 3), dtype=np.uint8),
        np.random.randint(0, 255, (1089, 1027, 3), dtype=np.uint8),
    ]
    with ds:
        ds.create_tensor(
            "x", **compression, max_chunk_size=1 * MB, tiling_threshold=1 * MB
        )
        ds.x.extend(arr_list)
    assert len(ds) == 2
    assert len(ds.x) == 2

    for i in range(2):
        np.testing.assert_array_equal(ds.x[i].numpy(), arr_list[i])


@compressions_paremetrized
def test_no_tiling(memory_ds, compression):
    ds = memory_ds
    arr_list = [
        np.random.randint(0, 255, (3894, 4279, 3), dtype=np.uint8),
        np.random.randint(0, 255, (1089, 1027, 3), dtype=np.uint8),
    ]
    with ds:
        ds.create_tensor("x", **compression, max_chunk_size=1 * MB, tiling_threshold=-1)
        ds.x.extend(arr_list)
    assert len(ds) == 2
    assert len(ds.x) == 2
    assert ds.x.chunk_engine.num_chunks == 2


def test_chunk_sizes(memory_ds):
    ds = memory_ds
    with ds:
        ds.create_tensor("image", htype="image", sample_compression="png")
        sample = np.random.randint(0, 256, size=(5000, 5000, 3), dtype=np.uint8)
        ds.image.append(sample)
    np.testing.assert_array_equal(ds.image[0].numpy(), sample)
    num_chunks = 0
    for k in ds.storage:
        if k.startswith("image/chunks/"):
            chunk = ds.storage[k].tobytes()
            num_chunks += 1
            assert 16 * MB < len(chunk) < 20 * MB
    assert num_chunks == 4
