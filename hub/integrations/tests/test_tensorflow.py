from hub.core.dataset import Dataset
from hub.util.check_installation import requires_tensorflow
import numpy as np
import hub
import pytest


@requires_tensorflow
def test_tensorflow_with_compression(local_ds: Dataset):
    # TODO: when chunk-wise compression is done, `labels` should be compressed using lz4, so this test needs to be updated
    images = local_ds.create_tensor("images", htype="image", sample_compression="png")
    labels = local_ds.create_tensor("labels", htype="class_label")

    images.extend(np.ones((16, 10, 10, 3), dtype="uint8"))
    labels.extend(np.ones((16, 1), dtype="uint32"))

    for batch in local_ds.tensorflow():
        # converting tf Tensors to numpy
        X = batch["images"].numpy()
        T = batch["labels"].numpy()
        assert X.shape == (10, 10, 3)
        assert T.shape == (1,)


@requires_tensorflow
def test_tensorflow_small(local_ds):
    local_ds.create_tensor("image")
    local_ds.image.extend(np.array([i * np.ones((10, 10)) for i in range(256)]))
    local_ds.create_tensor("image2")
    local_ds.image2.extend(np.array([i * np.ones((12, 12)) for i in range(256)]))
    local_ds.flush()

    tds = local_ds.tensorflow()
    for i, batch in enumerate(tds):
        # converting tf Tensors to numpy
        np.testing.assert_array_equal(batch["image"].numpy(), i * np.ones((10, 10)))
        np.testing.assert_array_equal(batch["image2"].numpy(), i * np.ones((12, 12)))


@requires_tensorflow
def test_corrupt_dataset(local_ds, corrupt_image_paths, compressed_image_paths):
    img_good = hub.read(compressed_image_paths["jpeg"][0])
    img_bad = hub.read(corrupt_image_paths["jpeg"])
    with local_ds:
        local_ds.create_tensor("image", htype="image", sample_compression="jpeg")
        for i in range(3):
            for i in range(10):
                local_ds.image.append(img_good)
            local_ds.image.append(img_bad)
    num_samples = 0
    with pytest.warns(UserWarning):
        tds = local_ds.tensorflow()
        for batch in tds:
            num_samples += 1  # batch_size = 1
    assert num_samples == 30
