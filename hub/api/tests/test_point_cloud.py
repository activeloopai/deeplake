import os
import sys
import pytest

import hub
from hub.core.dataset import Dataset
from hub.core.sample import Sample

import numpy as np


@pytest.mark.parametrize("compression", hub.compression.POINT_CLOUD_COMPRESSIONS)
def test_point_cloud(local_ds, point_cloud_paths, compression):
    for i, (compression, path) in enumerate(point_cloud_paths.items()):
        tensor = local_ds.create_tensor(f"point_cloud_{i}", htype="point_cloud", sample_compression=compression)
        sample = hub.read(path)
        if "dummy_data" in path:  # check shape only for internal test point_clouds
            if compression in ["las", "laz"]:
                assert sample.shape == (20153, 18)

        assert sample.shape[-1] == 18

        with local_ds:
            for _ in range(5):
                tensor.append(hub.read(path))  # type: ignore
            tensor.extend([hub.read(path) for _ in range(5)])  # type: ignore
        for i in range(10):
            assert tensor[i].numpy().shape[0] == sample.shape[0]  # type: ignore
            assert len(tensor[i].data()) == 6

        assert len(sample.meta["las_header"]) == 17
        assert type(sample.meta["dimension_names"]) == dict
        assert type(sample.meta["las_header"]) == dict
        assert type(sample.meta["las_header"]["DEFAULT_VERSION"]) == dict
        assert type(sample.meta["las_header"]["file_source_id"]) == int
        assert type(sample.meta["las_header"]["system_identifier"]) == str
        assert type(sample.meta["las_header"]["generating_software"]) == str
        assert type(sample.meta["las_header"]["creation_date"]) == dict
        assert type(sample.meta["las_header"]["point_count"]) == int
        assert type(sample.meta["las_header"]["scales"]) == np.ndarray
        assert type(sample.meta["las_header"]["offsets"]) == np.ndarray
        assert (
            type(sample.meta["las_header"]["number_of_points_by_return"]) == np.ndarray
        )
        assert (
            type(sample.meta["las_header"]["start_of_waveform_data_packet_record"])
            == int
        )
        assert type(sample.meta["las_header"]["start_of_first_evlr"]) == int
        assert type(sample.meta["las_header"]["number_of_evlrs"]) == int
        assert type(sample.meta["las_header"]["version"]) == dict
        assert type(sample.meta["las_header"]["maxs"]) == np.ndarray
        assert type(sample.meta["las_header"]["mins"]) == np.ndarray
        assert type(sample.meta["las_header"]["major_version"]) == int
        assert type(sample.meta["las_header"]["minor_version"]) == int


@pytest.mark.parametrize("compression", hub.compression.POINT_CLOUD_COMPRESSIONS)
def test_point_cloud_slicing(local_ds: Dataset, point_cloud_paths, compression):
    for compression, path in point_cloud_paths.items():
        if compression in ["las", "laz"]:
            dummy = np.zeros((20153, 3))

            local_ds.create_tensor("point_cloud", htype="point_cloud", sample_compression=compression)
            local_ds.point_cloud.append(hub.read(path))
            local_ds.point_cloud[0][0:5].numpy().shape == dummy[0:5].shape
            local_ds.point_cloud[0][100:120].numpy().shape == dummy[100:120].shape
            local_ds.point_cloud[0][120].numpy().shape == dummy[120].shape
            local_ds.point_cloud[0][-1].numpy().shape == dummy[-1].shape
            local_ds.point_cloud[0][10:5:-2].numpy().shape == dummy[10:5:-2].shape
            local_ds.point_cloud[0][-3:-10:-1].numpy().shape == dummy[-3:-10:-1].shape
            local_ds.point_cloud[0][-25:100:-2].numpy().shape == dummy[-25:100:-2].shape
            local_ds.point_cloud[0][::-1].numpy().shape == dummy[::-1].shape
            local_ds.point_cloud[0][:5:-1].numpy().shape == dummy[:5:-1].shape
            local_ds.point_cloud[0][-1].numpy().shape == dummy[-1].shape
            return
