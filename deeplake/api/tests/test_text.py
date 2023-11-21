import deeplake
import pytest
from deeplake.tests.dataset_fixtures import enabled_non_gcs_gdrive_datasets
import numpy as np


def test_text(memory_ds):
    ds = memory_ds
    ds.create_tensor("text", htype="text")
    items = ["abcd", "efgh", "0123456"]
    with ds:
        for x in items:
            ds.text.append(x)
        ds.text.extend(items)
    assert ds.text.shape == (6, 1)
    for i in range(6):
        assert ds.text[i].numpy()[0] == items[i % 3]


@pytest.mark.slow
@enabled_non_gcs_gdrive_datasets
def test_text_transform(ds, scheduler="threaded"):
    ds.create_tensor("text", htype="text")

    @deeplake.compute
    def upload(some_str, ds):
        ds.text.append(some_str)
        return ds

    upload().eval(
        ["hi", "if ur reading this ur a nerd"], ds, num_workers=2, scheduler=scheduler
    )

    assert len(ds) == 2
    assert ds.text.data()["value"] == ["hi", "if ur reading this ur a nerd"]


@pytest.mark.parametrize(
    "args", [{}, {"sample_compression": "lz4"}, {"chunk_compression": "lz4"}]
)
def test_text_update(memory_ds, args):
    ds = memory_ds
    with ds:
        ds.create_tensor("x", htype="text", **args)
        for _ in range(10):
            ds.x.append("cat")
        assert ds.x[0].text() == "cat"

    for i in range(0, 10, 2):
        ds.x[i] = "flower"
    assert ds.x.data()["value"] == ["flower", "cat"] * 5
    assert ds.x.text() == ["flower", "cat"] * 5


@pytest.mark.parametrize(
    "args", [{}, {"sample_compression": "lz4"}, {"chunk_compression": "lz4"}]
)
def test_extend_with_numpy(memory_ds, args):
    ds = memory_ds
    with ds:
        ds.create_tensor("x", htype="text", **args)
        ds.x.extend(["ab", "bcd", "cdefg"])
    ds2 = deeplake.empty("mem://")
    with ds2:
        ds2.create_tensor("x", htype="text", **args)
        ds2.x.extend(ds.x.numpy(aslist=True))
    np.testing.assert_array_equal(ds.x.numpy(), ds2.x.numpy())


@pytest.mark.parametrize(
    "args", [{}, {"sample_compression": "lz4"}, {"chunk_compression": "lz4"}]
)
def test_text_rechunk(memory_ds, args):
    ds = memory_ds
    with ds:
        ds.create_tensor("x", htype="text", max_chunk_size=16, **args)
        ds.x.extend(["abcd"] * 100)
        assert len(ds.x.chunk_engine.chunk_id_encoder.array) > 2
        ds.rechunk()
    assert ds.x.numpy().reshape(-1).tolist() == ["abcd"] * 100


def test_text_tensor_append(memory_ds):
    with memory_ds as ds:
        ds.create_tensor("x", htype="text", chunk_compression="lz4")
        ds.create_tensor("y", htype="json")
        ds.x.extend(["x", "y", "z"])
        ds.y.extend([{"a": "b"}, {"b": "c"}, {"c": "d"}])
        ds2 = deeplake.empty("mem://")
        with ds2:
            ds2.create_tensor("x", htype="text")
            ds2.create_tensor("y", htype="json", chunk_compression="lz4")
            ds2.x.extend(ds.x)
            ds2.y.extend(ds.y)
        for i in range(3):
            assert ds.x[i].data() == ds2.x[i].data()
            assert ds.y[i].data() == ds2.y[i].data()


def test_empty_text(memory_ds):
    with memory_ds as ds:
        ds.create_tensor("text", htype="text")

    assert ds.text.data()["value"] == []
