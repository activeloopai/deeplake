import numpy as np
import pytest


@pytest.mark.parametrize(
    "compression",
    [
        {"sample_compression": None},
        {"sample_compression": "lz4"},
        {"chunk_compression": "lz4"},
    ],
)
@pytest.mark.parametrize("insert_first", [True, False])
def test_insertion_array(memory_ds, compression, insert_first):
    with memory_ds as ds:
        ds.create_tensor("abc", **compression)
        first = np.random.rand(100, 100, 3)
        tenth = np.random.rand(100, 100, 3)
        empty_sample = np.random.rand(0, 0, 0)
        if insert_first:
            ds.abc.append(first)
        ds.abc.__setitem__(10, tenth, True)

        if insert_first:
            np.testing.assert_array_equal(ds.abc[0].numpy(), first)
        else:
            np.testing.assert_array_equal(ds.abc[0].numpy(), empty_sample)

        for i in range(1, 10):
            np.testing.assert_array_equal(ds.abc[i].numpy(), empty_sample)
        np.testing.assert_array_equal(ds.abc[10].numpy(), tenth)


@pytest.mark.parametrize("sample_compression", ["png", "jpeg"])
def test_insertion_array_png_jpeg(memory_ds, sample_compression):
    with memory_ds as ds:
        ds.create_tensor("abc", sample_compression=sample_compression)
        first = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)
        tenth = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        empty_sample = np.random.randint(0, 256, (0, 0, 0), dtype=np.uint8)
        ds.abc.append(first)
        ds.abc.__setitem__(10, tenth, True)

        if sample_compression == "png":
            np.testing.assert_array_equal(ds.abc[0].numpy(), first)
        else:
            assert ds.abc[0].numpy().shape == first.shape

        for i in range(1, 10):
            np.testing.assert_array_equal(ds.abc[i].numpy(), empty_sample)

        if sample_compression == "png":
            np.testing.assert_array_equal(ds.abc[10].numpy(), tenth)
        else:
            assert ds.abc[10].numpy().shape == tenth.shape
