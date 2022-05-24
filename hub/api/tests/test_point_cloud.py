import os
import sys
import pytest

import hub
from hub.core.dataset import Dataset

import numpy as np


@pytest.mark.skipif(
    os.name == "nt" and sys.version_info < (3, 7), reason="requires python 3.7 or above"
)
# @pytest.mark.parametrize("compression", hub.compression.POINT_CLOUD_COMPRESSIONS)
def test_point_cloud(local_ds, point_cloud_paths):
    for i, (compression, path) in enumerate(point_cloud_paths.items()):
        tensor = local_ds.create_tensor(
            f"point_cloud_{i}", htype="point_cloud"
        )
        sample = hub.read(path)
        if "dummy_data" in path:  # check shape only for internal test point_clouds
            if compression == "las":
                assert sample.shape == (20153, 3)

        assert sample.shape[-1] == 3

        with local_ds:
            for _ in range(5):
                tensor.append(hub.read(path))  # type: ignore
            tensor.extend([hub.read(path) for _ in range(5)])  # type: ignore
        for i in range(10):
            assert tensor[i].numpy().shape == sample.shape  # type: ignore


@pytest.mark.skipif(
    os.name == "nt" and sys.version_info < (3, 7), reason="requires python 3.7 or above"
)
def test_point_cloud_slicing(local_ds: Dataset, point_cloud_paths):
    for compression, path in point_cloud_paths.items():
        if compression == "las":
            dummy = np.zeros((20153, 3))

            local_ds.create_tensor("point_cloud", htype="point_cloud")
            local_ds.point_cloud.append(hub.read(path))
            local_ds.point_cloud[0][0:5].numpy().shape == dummy[0:5].shape
            local_ds.point_cloud[0][100:120].numpy().shape == dummy[100:120].shape
            local_ds.point_cloud[0][120].numpy().shape == dummy[120].shape
            local_ds.point_cloud[0][-1].numpy().shape == dummy[-1].shape
            return
    raise Exception  # test did not run
