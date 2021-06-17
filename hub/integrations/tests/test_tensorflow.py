import numpy as np

from hub.util.check_installation import requires_tensorflow


# TODO: separate test with compression


@requires_tensorflow
def test_tensorflow_small(local_ds):
    local_ds.create_tensor("image")
    local_ds.image.extend(np.array([i * np.ones((300, 300)) for i in range(256)]))
    local_ds.create_tensor("image2")
    local_ds.image2.extend(np.array([i * np.ones((100, 100)) for i in range(256)]))
    local_ds.flush()

    tds = local_ds.tensorflow()
    for i, batch in enumerate(tds):
        # converting tf Tensors to numpy
        np.testing.assert_array_equal(batch["image"].numpy(), i * np.ones((300, 300)))
        np.testing.assert_array_equal(batch["image2"].numpy(), i * np.ones((100, 100)))
    local_ds.delete()


@requires_tensorflow
def test_tensorflow_large(local_ds):
    local_ds.create_tensor("image")
    arr = np.array(
        [
            np.ones((4096, 4096)),
            2 * np.ones((4096, 4096)),
            3 * np.ones((4096, 4096)),
            4 * np.ones((4096, 4096)),
        ],
    )
    local_ds.image.extend(arr)
    local_ds.create_tensor("classlabel")
    local_ds.classlabel.extend(np.array([i for i in range(10)]))
    local_ds.flush()

    tds = local_ds.tensorflow()
    for i, batch in enumerate(tds):
        # converting tf Tensors to numpy
        np.testing.assert_array_equal(
            batch["image"].numpy(), (i + 1) * np.ones((4096, 4096))
        )
        np.testing.assert_array_equal(batch["classlabel"].numpy(), (i) * np.ones((1,)))
    local_ds.delete()
