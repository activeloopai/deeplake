import hub
from hub.core.dataset import Dataset

import numpy as np


def test_point_cloud(local_ds, point_cloud_paths):
    for i, (compression, path) in enumerate(point_cloud_paths.items()):
        if compression == "las":
            tensor = local_ds.create_tensor(
                f"point_cloud_{i}", htype="point_cloud", sample_compression=compression
            )
            sample = hub.read(path)
            if "dummy_data" in path:  # check shape only for internal test point_clouds
                assert sample.shape[0] == 20153

            assert len(sample.meta) == 6
            assert len(sample.meta["dimension_names"]) == 18
            assert len(sample.meta["las_header"]) == 23
            tensor.append(sample)
            tensor.append(sample)
            tensor.append(sample)
            assert tensor.shape == (3, 20153)

            shape_tester(local_ds, path, sample, tensor, feature_size=18)


def shape_tester(local_ds, path, sample, tensor, feature_size):
    with local_ds:
        for _ in range(5):
            tensor.append(hub.read(path))  # type: ignore
        tensor.extend([hub.read(path) for _ in range(5)])  # type: ignore

    for i in range(10):
        assert tensor[i].numpy().shape[0] == sample.shape[0]  # type: ignore
        assert len(tensor[i].data()) == feature_size


def test_point_cloud_slicing(local_ds: Dataset, point_cloud_paths):
    for compression, path in point_cloud_paths.items():
        if compression == "las":
            dummy = np.zeros((20153, 3))
        local_ds.create_tensor(
            "point_cloud", htype="point_cloud", sample_compression=compression
        )
        local_ds.point_cloud.append(hub.read(path))
        assert local_ds.point_cloud[0][0:5].numpy().shape == dummy[0:5].shape
        assert local_ds.point_cloud[0][100:120].numpy().shape == dummy[100:120].shape
        assert local_ds.point_cloud[0][120].numpy().shape == dummy[120].shape
        assert local_ds.point_cloud[0][-1].numpy().shape == dummy[-1].shape
        assert local_ds.point_cloud[0][10:5:-2].numpy().shape == dummy[10:5:-2].shape
        assert (
            local_ds.point_cloud[0][-3:-10:-1].numpy().shape == dummy[-3:-10:-1].shape
        )
        assert (
            local_ds.point_cloud[0][-25:100:-2].numpy().shape == dummy[-25:100:-2].shape
        )
        assert local_ds.point_cloud[0][::-1].numpy().shape == dummy[::-1].shape
        assert local_ds.point_cloud[0][:5:-1].numpy().shape == dummy[:5:-1].shape
        assert local_ds.point_cloud[0][-1].numpy().shape == dummy[-1].shape
        return
