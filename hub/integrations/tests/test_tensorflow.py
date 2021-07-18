from hub.api.dataset import Dataset
import numpy as np

from hub.util.check_installation import requires_tensorflow


@requires_tensorflow
def test_tensorflow_with_compression(local_ds: Dataset):
    # TODO: when chunk-wise compression is done, `labels` should be compressed using lz4, so this test needs to be updated
    images = local_ds.create_tensor("images", htype="image", sample_compression="png")
    labels = local_ds.create_tensor("labels", htype="class_label")

    images.extend(np.ones((16, 100, 100, 3), dtype="uint8"))
    labels.extend(np.ones((16, 1), dtype="uint32"))

    for batch in local_ds.tensorflow():
        # converting tf Tensors to numpy
        X = batch["images"].numpy()
        T = batch["labels"].numpy()
        assert X.shape == (100, 100, 3)
        assert T.shape == (1,)


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
