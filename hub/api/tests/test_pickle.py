import pytest

import pickle
from numpy import (
    uint8,
    ones as np_ones,
    testing as np_testing
)

from hub.util.exceptions import MemoryDatasetCanNotBePickledError


@pytest.mark.parametrize(
    "ds",
    ["memory_ds", "local_ds", "s3_ds", "gcs_ds", "hub_cloud_ds"],
    indirect=True,
)
def test_dataset(ds):
    if ds.path.startswith("mem://"):
        with pytest.raises(MemoryDatasetCanNotBePickledError):
            pickle.dumps(ds)
        return

    with ds:
        ds.create_tensor("image", htype="image", sample_compression="jpeg")
        ds.create_tensor("label")
        for i in range(10):
            ds.image.append(
                i * np_ones(((i + 1) * 20, (i + 1) * 20, 3), dtype=uint8)
            )

        for i in range(5):
            ds.label.append(i)

    pickled_ds = pickle.dumps(ds)
    unpickled_ds = pickle.loads(pickled_ds)
    assert len(unpickled_ds.image) == len(ds.image)
    assert len(unpickled_ds.label) == len(ds.label)
    assert unpickled_ds.tensors.keys() == ds.tensors.keys()
    assert unpickled_ds.index.values[0].value == ds.index.values[0].value
    assert unpickled_ds.meta.version == ds.meta.version

    for i in range(10):
        np_testing.assert_array_equal(
            ds.image[i].numpy(),
            (i * np_ones(((i + 1) * 20, (i + 1) * 20, 3), dtype=uint8)),
        )
        np_testing.assert_array_equal(
            ds.image[i].numpy(), unpickled_ds.image[i].numpy()
        )
    for i in range(5):
        np_testing.assert_array_equal(ds.label[i].numpy(), i)
        np_testing.assert_array_equal(
            ds.label[i].numpy(), unpickled_ds.label[i].numpy()
        )
