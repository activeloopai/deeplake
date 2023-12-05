import numpy as np
import pytest
from deeplake.util.exceptions import MemoryDatasetCanNotBePickledError
import pickle


@pytest.mark.parametrize(
    "ds",
    [
        "memory_ds",
        "local_ds",
        pytest.param("s3_ds", marks=pytest.mark.slow),
        pytest.param("gcs_ds", marks=pytest.mark.slow),
        pytest.param("hub_cloud_ds", marks=pytest.mark.slow),
    ],
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


def test_pickled_dataset_does_not_store_cached_chunks(local_ds):
    with local_ds as ds:
        ds.create_tensor("abc")
        ds.abc.append([1, 2, 3, 4, 5])

    size = len(pickle.dumps(ds))

    ds.abc.append([1, 2, 3, 4, 5])

    assert (
        len(pickle.dumps(ds)) == size
    ), "Adding element should not change size of pickle"
