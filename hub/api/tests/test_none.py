import numpy as np
import pytest

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
            with pytest.raises(ValueError):
                ds.xyz[i].numpy()
            assert ds.xyz[i].shape is None


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
