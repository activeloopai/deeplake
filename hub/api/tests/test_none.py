import numpy as np
import pytest
from hub.util.exceptions import EmptyTensorError

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
def test_none_append(local_ds, compression, create_shape_tensor):
    with local_ds as ds:
        ds.create_tensor("xyz", create_shape_tensor=create_shape_tensor, **compression)
        ds.xyz.append(None)
        ds.xyz.append(None)
        ds.xyz.append(np.ones((100, 100, 3), dtype=np.uint8))

        for i in range(2):
            assert ds.xyz[i].numpy().shape == (0, 0, 0)
            assert ds.xyz[i].shape == (0, 0, 0)
        assert ds.xyz[2].numpy().shape == (100, 100, 3)
        assert ds.xyz[2].shape == (100, 100, 3)


@compressions_paremetrized
@pytest.mark.parametrize("create_shape_tensor", [True, False])
def test_only_nones_append(local_ds, compression, create_shape_tensor):
    with local_ds as ds:
        ds.create_tensor("xyz", create_shape_tensor=create_shape_tensor, **compression)
        ds.xyz.append(None)
        ds.xyz.append(None)

        for i in range(2):
            with pytest.raises(EmptyTensorError):
                ds.xyz[i].numpy()
            assert ds.xyz[i].shape == ()


@compressions_paremetrized
@pytest.mark.parametrize("create_shape_tensor", [True, False])
def test_none_updates(local_ds, compression, create_shape_tensor):
    with local_ds as ds:
        ds.create_tensor("xyz", **compression)
        ds.xyz.append(np.ones((100, 100, 3), dtype=np.uint8))
        ds.xyz.append(np.ones((300, 500, 3), dtype=np.uint8))
        ds.xyz.append(np.ones((300, 500, 3), dtype=np.uint8))

        ds.xyz[1] = None
        assert ds.xyz[0].numpy().shape == (100, 100, 3)
        assert ds.xyz[0].shape == (100, 100, 3)
        assert ds.xyz[1].numpy().shape == (0, 0, 0)
        assert ds.xyz[1].shape == (0, 0, 0)
        assert ds.xyz[2].numpy().shape == (300, 500, 3)
        assert ds.xyz[2].shape == (300, 500, 3)


def test_none_image_chunk_compression_2d(local_ds):
    with local_ds as ds:
        ds.create_tensor("xyz", chunk_compression="png")
        ds.xyz.append(None)
        ds.xyz.append(None)
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


def test_none_text(local_ds):
    with local_ds as ds:
        ds.create_tensor("xyz", htype="text")
        ds.xyz.append(None)
        ds.xyz.append(None)
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


def test_none_json(local_ds):
    with local_ds as ds:
        ds.create_tensor("xyz", htype="json")
        ds.xyz.append(None)
        ds.xyz.append(None)
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


def test_none_list(local_ds):
    with local_ds as ds:
        ds.create_tensor("xyz", htype="list")
        ds.xyz.append(None)
        ds.xyz.append(None)
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
