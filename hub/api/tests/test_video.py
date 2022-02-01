import pytest
from hub.tests.dataset_fixtures import enabled_datasets

import hub
from hub.core.dataset import Dataset

import os

if os.name == "nt":
    _USE_CFFI = False
else:
    _USE_CFFI = True


@pytest.mark.parametrize("compression", hub.compression.VIDEO_COMPRESSIONS)
def test_video(local_ds, compression, video_paths):
    for i, path in enumerate(video_paths[compression]):
        tensor = local_ds.create_tensor(
            f"video_{i}", htype="video", sample_compression=compression
        )
        sample = hub.read(path)
        assert len(sample.shape) == 4
        if "dummy_data" in path:  # check shape only for internal test videos
            if compression in ("mp4", "mkv"):
                if (
                    _USE_CFFI
                ):  # cffi and slower implementation outputs different number of frames
                    assert sample.shape == (377, 360, 640, 3)
                else:
                    assert sample.shape == (400, 360, 640, 3)
            elif compression == "avi":
                assert sample.shape == (900, 270, 480, 3)
        assert sample.shape[-1] == 3
        with local_ds:
            for _ in range(5):
                tensor.append(hub.read(path))  # type: ignore
            tensor.extend([hub.read(path) for _ in range(5)])  # type: ignore
        for i in range(10):
            assert tensor[i].numpy().shape == sample.shape  # type: ignore
