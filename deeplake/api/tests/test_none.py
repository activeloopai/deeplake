import numpy as np
import pytest
from deeplake.util.exceptions import EmptyTensorError

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


@compressions_paremetrized
@pytest.mark.parametrize("create_shape_tensor", [True, False])
@pytest.mark.parametrize("empty_sample", [None, []])
def test_none_append(local_ds, compression, create_shape_tensor, empty_sample):
    with local_ds as ds:
        ds.create_tensor("xyz", create_shape_tensor=create_shape_tensor, **compression)
        ds.xyz.append(empty_sample)
        ds.xyz.append(empty_sample)
        ds.xyz.append(np.ones((100, 100, 3), dtype=np.uint8))

        for i in range(2):
            assert ds.xyz[i].numpy().shape == (0, 0, 0)
            assert ds.xyz[i].shape == (0, 0, 0)
        assert ds.xyz[2].numpy().shape == (100, 100, 3)
        assert ds.xyz[2].shape == (100, 100, 3)


@compressions_paremetrized
@pytest.mark.parametrize("create_shape_tensor", [True, False])
@pytest.mark.parametrize("empty_sample", [None, []])
def test_only_nones_append(local_ds, compression, create_shape_tensor, empty_sample):
    with local_ds as ds:
        ds.create_tensor("xyz", create_shape_tensor=create_shape_tensor, **compression)
        ds.xyz.append(empty_sample)
        ds.xyz.append(empty_sample)

        for i in range(2):
            with pytest.raises(EmptyTensorError):
                ds.xyz[i].numpy()
            assert ds.xyz[i].shape == ()


@compressions_paremetrized
@pytest.mark.parametrize("create_shape_tensor", [True, False])
@pytest.mark.parametrize("empty_sample", [None, []])
def test_none_updates(local_ds, compression, create_shape_tensor, empty_sample):
    with local_ds as ds:
        ds.create_tensor("xyz", create_shape_tensor=create_shape_tensor, **compression)
        ds.xyz.append(np.ones((100, 100, 3), dtype=np.uint8))
        ds.xyz.append(np.ones((300, 500, 3), dtype=np.uint8))
        ds.xyz.append(np.ones((300, 500, 3), dtype=np.uint8))

        ds.xyz[1] = empty_sample
        assert ds.xyz[0].numpy().shape == (100, 100, 3)
        assert ds.xyz[0].shape == (100, 100, 3)
        assert ds.xyz[1].numpy().shape == (0, 0, 0)
        assert ds.xyz[1].shape == (0, 0, 0)
        assert ds.xyz[2].numpy().shape == (300, 500, 3)
        assert ds.xyz[2].shape == (300, 500, 3)


@pytest.mark.parametrize("empty_sample", [None, []])
def test_none_image_chunk_compression_2d(local_ds, empty_sample):
    with local_ds as ds:
        ds.create_tensor("xyz", chunk_compression="png")
        ds.xyz.append(empty_sample)
        ds.xyz.append(empty_sample)
        assert ds.xyz.meta.max_shape == [0, 0, 0]
        assert ds.xyz[0].shape == ()
        assert ds.xyz[1].shape == ()
        ds.xyz.append(np.ones((500, 500), "uint8"))
        assert ds.xyz.meta.max_shape == [500, 500]
        assert ds.xyz[0].numpy().shape == (0, 0)
        assert ds.xyz[0].shape == (0, 0)
        assert ds.xyz[1].numpy().shape == (0, 0)
        assert ds.xyz[1].shape == (0, 0)
        assert ds.xyz[2].numpy().shape == (500, 500)
        assert ds.xyz[2].shape == (500, 500)


@pytest.mark.parametrize("empty_sample", [None, []])
def test_none_text(local_ds, empty_sample):
    with local_ds as ds:
        ds.create_tensor("xyz", htype="text")
        ds.xyz.append(empty_sample)
        ds.xyz.append(empty_sample)
        assert ds.xyz.meta.max_shape == [1]
        assert ds.xyz[0].numpy().shape == (1,)
        assert ds.xyz[0].shape == (1,)
        assert ds.xyz[0].numpy() == ""
        assert ds.xyz[1].numpy().shape == (1,)
        assert ds.xyz[1].shape == (1,)
        assert ds.xyz[1].numpy() == ""

        ds.xyz.append("hello")
        assert ds.xyz.meta.max_shape == [1]
        assert ds.xyz[0].numpy().shape == (1,)
        assert ds.xyz[0].shape == (1,)
        assert ds.xyz[0].numpy() == ""
        assert ds.xyz[1].numpy().shape == (1,)
        assert ds.xyz[1].shape == (1,)
        assert ds.xyz[1].numpy() == ""
        assert ds.xyz[2].numpy().shape == (1,)
        assert ds.xyz[2].shape == (1,)
        assert ds.xyz[2].numpy() == "hello"


@pytest.mark.parametrize("empty_sample", [None, []])
def test_none_json(local_ds, empty_sample):
    with local_ds as ds:
        ds.create_tensor("xyz", htype="json")
        ds.xyz.append(empty_sample)
        ds.xyz.append(empty_sample)
        assert ds.xyz.meta.max_shape == [1]
        assert ds.xyz[0].numpy().shape == (1,)
        assert ds.xyz[0].shape == (1,)
        assert ds.xyz[0].numpy() == {}
        assert ds.xyz[1].numpy().shape == (1,)
        assert ds.xyz[1].shape == (1,)
        assert ds.xyz[1].numpy() == {}

        ds.xyz.append({"hello": "world"})
        assert ds.xyz.meta.max_shape == [1]
        assert ds.xyz[0].numpy().shape == (1,)
        assert ds.xyz[0].shape == (1,)
        assert ds.xyz[0].numpy() == {}
        assert ds.xyz[1].numpy().shape == (1,)
        assert ds.xyz[1].shape == (1,)
        assert ds.xyz[1].numpy() == {}
        assert ds.xyz[2].numpy().shape == (1,)
        assert ds.xyz[2].shape == (1,)
        assert ds.xyz[2].numpy() == {"hello": "world"}


@pytest.mark.parametrize("empty_sample", [None, []])
def test_none_list(local_ds, empty_sample):
    with local_ds as ds:
        ds.create_tensor("xyz", htype="list")
        ds.xyz.append(empty_sample)
        ds.xyz.append(empty_sample)
        assert ds.xyz.meta.max_shape == [0]
        assert ds.xyz[0].numpy().shape == (0,)
        assert ds.xyz[0].shape == (0,)
        assert ds.xyz[0].numpy().tolist() == []
        assert ds.xyz[1].numpy().shape == (0,)
        assert ds.xyz[1].shape == (0,)
        assert ds.xyz[1].numpy().tolist() == []

        ds.xyz.append(["hello", "world"])
        assert ds.xyz.meta.max_shape == [2]
        assert ds.xyz[0].numpy().shape == (0,)
        assert ds.xyz[0].shape == (0,)
        assert ds.xyz[0].numpy().tolist() == []
        assert ds.xyz[1].numpy().shape == (0,)
        assert ds.xyz[1].shape == (0,)
        assert ds.xyz[1].numpy().tolist() == []
        assert ds.xyz[2].numpy().shape == (2,)
        assert ds.xyz[2].shape == (2,)
        assert ds.xyz[2].numpy().tolist() == ["hello", "world"]


def test_none_bugs(local_ds):
    with local_ds as ds:
        ds.create_tensor("abc")
        ds.abc.extend(
            [
                None,
                np.array([80, 22, 1]),
                None,
                np.array([0, 565, 234]),
            ]
        )

        ds.create_tensor("xyz", dtype="int64")
        ds.xyz.extend(
            [
                None,
                np.array([80, 22, 1]),
                None,
                np.array([0, 565, 234]),
            ]
        )

    assert ds.abc.htype == "generic"
    assert ds.xyz.htype == "generic"
    assert ds.xyz.dtype == np.dtype("int64")

    with local_ds as ds:
        ds.create_tensor("dummy1")
        ds.dummy1.extend(
            np.array(
                [None, np.array([80, 22, 1]), None, np.array([0, 565, 234])],
                dtype=object,
            )
        )

        ds.create_tensor("dummy2", dtype="int64")
        ds.dummy2.extend(
            np.array(
                [None, np.array([80, 22, 1]), None, np.array([0, 565, 234])],
                dtype=object,
            )
        )

    expected = [
        np.array([]),
        np.array([80, 22, 1]),
        np.array([]),
        np.array([0, 565, 234]),
    ]
    res = ds.dummy1.numpy(aslist=True)

    for i in range(len(expected)):
        np.testing.assert_array_equal(expected[i], res[i])

    assert ds.dummy2.dtype == np.dtype("int64")
    res = ds.dummy2.numpy(aslist=True)

    for i in range(len(expected)):
        np.testing.assert_array_equal(expected[i], res[i])
