import pytest
from hub.tests.dataset_fixtures import enabled_datasets

import hub
from hub.core.dataset import Dataset

import os
import numpy as np

from hub.core.compression import _decompress_video_pipes

if os.name == "nt":
    _USE_CFFI = False
else:
    _USE_CFFI = True


@pytest.mark.parametrize("compression", hub.compression.VIDEO_COMPRESSIONS)
def test_video(local_ds, compression, video_paths):
    for i, path in enumerate(video_paths[compression]):
        if "big_buck_bunny" in path:
            continue
        tensor = local_ds.create_tensor(
            f"video_{i}", htype="video", sample_compression=compression
        )
        sample = hub.read(path)
        assert len(sample.shape) == 4
        if "dummy_data" in path:  # check shape only for internal test videos
            if compression == "mp4":
                assert sample.shape == (400, 360, 640, 3)
            elif compression == "mkv":
                if _USE_CFFI:
                    assert sample.shape == (399, 360, 640, 3)
                else:
                    assert sample.shape == (400, 360, 640, 3)
            elif compression == "avi":
                if _USE_CFFI:
                    assert sample.shape == (901, 270, 480, 3)
                else:
                    assert sample.shape == (900, 270, 480, 3)
        assert sample.shape[-1] == 3
        with local_ds:
            for _ in range(5):
                tensor.append(hub.read(path))  # type: ignore
            tensor.extend([hub.read(path) for _ in range(5)])  # type: ignore
        for i in range(10):
            assert tensor[i].numpy().shape == sample.shape  # type: ignore


def test_video_slicing(local_ds: Dataset, video_paths):
    for path in video_paths["mp4"]:
        if "big_buck_bunny" in path:
            raw_video = _decompress_video_pipes(path, "mp4")
            assert raw_video.shape == (132, 720, 1280, 3)

            local_ds.create_tensor("video", htype="video", sample_compression="mp4")
            local_ds.video.append(hub.read(path))

            np.testing.assert_array_equal(local_ds.video[0][0:5], raw_video[0:5])
            np.testing.assert_array_equal(
                local_ds.video[0][100:120].numpy(), raw_video[100:120]
            )
            np.testing.assert_array_equal(
                local_ds.video[0][120].numpy(), raw_video[120]
            )
            np.testing.assert_array_equal(
                local_ds.video[0][10:5:-1].numpy(), raw_video[10:5:-1]
            )
            np.testing.assert_array_equal(
                local_ds.video[0][-3:-10:-1].numpy(), raw_video[-3:-10:-1]
            )
            np.testing.assert_array_equal(
                local_ds.video[0][-25:100:-1].numpy(), raw_video[-25:100:-1]
            )
            np.testing.assert_array_equal(
                local_ds.video[0][100:-25:-1].numpy(), raw_video[100:-25:-1]
            )
