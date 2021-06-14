from hub.util.check_installation import tensorflow_installed
import numpy as np
import pytest


requires_tensorflow = pytest.mark.skipif(
    not tensorflow_installed(), reason="requires tensorflow to be installed"
)


@requires_tensorflow
def test_pytorch_small_old(local_ds):
    local_ds.create_tensor("image")
    local_ds.image.extend(np.array([i * np.ones((300, 300)) for i in range(256)]))
    local_ds.create_tensor("image2")
    local_ds.image2.extend(np.array([i * np.ones((100, 100)) for i in range(256)]))
    local_ds.flush()

    tds = local_ds.tensorflow()
    for i, batch in enumerate(tds):
        np.testing.assert_array_equal(batch["image"].numpy(), i * np.ones((300, 300)))
        np.testing.assert_array_equal(batch["image2"].numpy(), i * np.ones((100, 100)))
    local_ds.delete()


@requires_tensorflow
def test_pytorch_large_old(local_ds):
    local_ds.create_tensor("image")
    arr = np.array(
        [
            np.ones((4096, 4096)),
            2 * np.ones((4096, 4096)),
            3 * np.ones((4096, 4096)),
            4 * np.ones((4096, 4096)),
        ]
    )
    local_ds.image.extend(arr)
    local_ds.create_tensor("classlabel")
    local_ds.classlabel.extend(np.array([i for i in range(10)]))
    local_ds.flush()

    tds = local_ds.tensorflow()
    for i, batch in enumerate(tds):
        np.testing.assert_array_equal(
            batch["image"].numpy(), (i + 1) * np.ones((4096, 4096))
        )
        np.testing.assert_array_equal(batch["classlabel"].numpy(), (i) * np.ones((1,)))
    local_ds.delete()
