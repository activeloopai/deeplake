import numpy as np

from hub.integrations.pytorch_old import dataset_to_pytorch
from hub.util.check_installation import requires_torch


@requires_torch
def test_pytorch_small(local_ds):
    import torch

    local_ds.create_tensor("image")
    local_ds.image.extend(np.array([i * np.ones((300, 300)) for i in range(256)]))
    local_ds.create_tensor("image2")
    local_ds.image2.extend(np.array([i * np.ones((100, 100)) for i in range(256)]))
    local_ds.flush()

    ptds = local_ds.pytorch(workers=2)

    # always use num_workers=0, when using hub workers
    dl = torch.utils.data.DataLoader(
        ptds,
        batch_size=1,
        num_workers=0,
    )
    for i, batch in enumerate(dl):
        np.testing.assert_array_equal(
            batch["image"].numpy(), i * np.ones((1, 300, 300))
        )
        np.testing.assert_array_equal(
            batch["image2"].numpy(), i * np.ones((1, 100, 100))
        )
    local_ds.delete()


@requires_torch
def test_pytorch_small_old(local_ds):
    import torch

    local_ds.create_tensor("image")
    local_ds.image.extend(np.array([i * np.ones((300, 300)) for i in range(256)]))
    local_ds.create_tensor("image2")
    local_ds.image2.extend(np.array([i * np.ones((100, 100)) for i in range(256)]))
    local_ds.flush()

    # .pytorch will automatically switch depending on version, this syntax is being used to ensure testing of old code on Python 3.8
    ptds = dataset_to_pytorch(local_ds, workers=2, python_version_warning=False)
    dl = torch.utils.data.DataLoader(
        ptds,
        batch_size=1,
        num_workers=0,
    )
    for i, batch in enumerate(dl):
        np.testing.assert_array_equal(
            batch["image"].numpy(), i * np.ones((1, 300, 300))
        )
        np.testing.assert_array_equal(
            batch["image2"].numpy(), i * np.ones((1, 100, 100))
        )
    local_ds.delete()


@requires_torch
def test_pytorch_large_old(local_ds):
    import torch

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
    local_ds.create_tensor("classlabel", htype="class_label")
    local_ds.classlabel.extend(np.array([i for i in range(10)], dtype="uint32"))
    local_ds.flush()

    # .pytorch will automatically switch depending on version, this syntax is being used to ensure testing of old code on Python 3.8
    ptds = dataset_to_pytorch(local_ds, workers=2, python_version_warning=False)
    dl = torch.utils.data.DataLoader(
        ptds,
        batch_size=1,
        num_workers=0,
    )
    for i, batch in enumerate(dl):
        actual_image = batch["image"].numpy()
        expected_image = (i + 1) * np.ones((1, 4096, 4096))

        actual_label = batch["classlabel"].numpy()
        expected_label = (i) * np.ones((1,))

        np.testing.assert_array_equal(actual_image, expected_image)
        np.testing.assert_array_equal(actual_label, expected_label)

    local_ds.delete()
