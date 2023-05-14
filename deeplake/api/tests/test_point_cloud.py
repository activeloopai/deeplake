import pytest

import deeplake
from deeplake.core.dataset import Dataset
from deeplake.core.compression import compress_multiple
from deeplake.tests.common import get_dummy_data_path
from deeplake.util.exceptions import SampleAppendError
import numpy as np


def test_point_cloud(local_ds, point_cloud_paths):
    for i, (compression, path) in enumerate(point_cloud_paths.items()):
        if compression == "las":
            tensor = local_ds.create_tensor(
                f"point_cloud_{i}", htype="point_cloud", sample_compression=compression
            )
            sample = deeplake.read(path)

            if "point_cloud" in path:  # check shape only for internal test point_clouds
                assert sample.shape[0] == 20153

            assert len(sample.meta) == 7
            assert len(sample.meta["dimension_names"]) == 18
            assert len(sample.meta["las_header"]) == 19

            assert sample.meta["las_header"]["DEFAULT_VERSION"] == {
                "major": 1,
                "minor": 2,
            }
            assert sample.meta["las_header"]["creation_date"] == {
                "year": 2022,
                "month": 5,
                "day": 24,
            }
            assert sample.meta["las_header"]["version"] == {"major": 1, "minor": 2}
            assert (
                sample.meta["las_header"]["uuid"]
                == "00000000-0000-0000-0000-000000000000"
            )

            tensor.append(sample)
            tensor.append(sample)
            tensor.append(sample)
            assert tensor.shape == (3, 20153, 18)

            shape_tester(local_ds, path, sample, tensor, feature_size=18)

            with pytest.raises(NotImplementedError):
                arrays = np.zeros((5, 1000, 3))
                compress_multiple(arrays, compression)

    local_ds.create_tensor(
        "point_cloud_without_sample_compression",
        htype="point_cloud",
        sample_compression=None,
    )
    local_ds.point_cloud_without_sample_compression.append(
        np.zeros((1000, 3), dtype=np.float32)
    )
    np.testing.assert_array_equal(
        local_ds.point_cloud_without_sample_compression[0].numpy(),
        np.zeros((1000, 3), dtype=np.float32),
    )
    local_ds.point_cloud_without_sample_compression.data()
    assert len(local_ds.point_cloud_without_sample_compression.data()) == 0

    local_ds.point_cloud_without_sample_compression.append(deeplake.read(path))
    assert local_ds.point_cloud_without_sample_compression[1].numpy().shape == (
        20153,
        3,
    )

    assert isinstance(
        local_ds.point_cloud_without_sample_compression.numpy(aslist=True), list
    )

    assert len(local_ds.point_cloud_without_sample_compression.numpy(aslist=True)) == 2
    assert len(local_ds.point_cloud_without_sample_compression.data(aslist=True)) == 2
    local_ds.create_tensor(
        "point_cloud_with_sample_compression",
        htype="point_cloud",
        sample_compression="las",
    )
    with pytest.raises(SampleAppendError):
        local_ds.point_cloud_with_sample_compression.append(
            np.zeros((1000, 3), dtype=np.float32)
        )

    with pytest.raises(SampleAppendError):
        local_ds.point_cloud_with_sample_compression.append(
            deeplake.read(get_dummy_data_path("point_cloud/corrupted_point_cloud.las"))
        )

    local_ds.point_cloud_with_sample_compression.append(
        deeplake.read(path, verify=True)
    )
    assert local_ds.point_cloud_with_sample_compression.shape == (1, 20153, 18)

    local_ds.create_tensor(
        "point_cloud_data_method_type_tester",
        htype="point_cloud",
        sample_compression="las",
    )
    local_ds.point_cloud_data_method_type_tester.append(sample)
    assert isinstance(local_ds.point_cloud_data_method_type_tester.data(), dict)

    local_ds.point_cloud_data_method_type_tester.append(sample)
    assert isinstance(
        local_ds.point_cloud_data_method_type_tester.data(aslist=True), list
    )


def shape_tester(local_ds, path, sample, tensor, feature_size):
    with local_ds:
        for _ in range(5):
            tensor.append(deeplake.read(path))  # type: ignore
        tensor.extend([deeplake.read(path) for _ in range(5)])  # type: ignore

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
        local_ds.point_cloud.append(deeplake.read(path))
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
