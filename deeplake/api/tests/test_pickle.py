import numpy as np
import pytest
from deeplake.util.exceptions import MemoryDatasetCanNotBePickledError
import pickle


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
                i * np.ones(((i + 1) * 20, (i + 1) * 20, 3), dtype=np.uint8)
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
        np.testing.assert_array_equal(
            ds.image[i].numpy(),
            (i * np.ones(((i + 1) * 20, (i + 1) * 20, 3), dtype=np.uint8)),
        )
        np.testing.assert_array_equal(
            ds.image[i].numpy(), unpickled_ds.image[i].numpy()
        )
    for i in range(5):
        np.testing.assert_array_equal(ds.label[i].numpy(), i)
        np.testing.assert_array_equal(
            ds.label[i].numpy(), unpickled_ds.label[i].numpy()
        )
