import hub
import pytest
import numpy as np


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_shape": (1003, 1103, 3), "tile_shape": (10, 10, 3)},
        {"sample_shape": (100003, 300007, 3)},
    ],
)
def test_partial_upload(memory_ds, kwargs):
    ds = memory_ds
    ds.create_tensor("image", htype="image", sample_compression="png")
    ds.image.append(hub.tiled(**kwargs))
    np.testing.assert_array_equal(
        ds.image[0][:10, :10].numpy(), np.zeros((10, 10, 3), dtype=np.uint8)
    )
    r = np.random.randint(0, 256, (217, 212, 2), dtype=np.uint8)
    ds.image[0][-217:, :212, 1:] = r
    np.testing.assert_array_equal(ds.image[0][-217:, :212, 1:].numpy(), r)
