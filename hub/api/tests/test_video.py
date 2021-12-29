import pytest
from hub.tests.dataset_fixtures import enabled_datasets

import hub
from hub.core.dataset import Dataset


@enabled_datasets
@pytest.mark.parametrize("compression", hub.compression.VIDEO_COMPRESSIONS)
def test_video(ds: Dataset, compression, video_paths):
    for i, path in enumerate(video_paths[compression]):
        tensor = ds.create_tensor(
            f"video_{i}", htype="video", sample_compression=compression
        )
        sample = hub.read(path)
        assert len(sample.shape) == 4
        if "dummy_data" in path:  # check shape only for internal test videos
            if compression in ("mp4", "mkv"):
                assert sample.shape == (400, 360, 640, 3)
            elif compression == "avi":
                assert sample.shape == (900, 270, 480, 3)
        assert sample.shape[-1] == 3
        with ds:
            for _ in range(5):
                tensor.append(hub.read(path))  # type: ignore
            tensor.extend([hub.read(path) for _ in range(5)])  # type: ignore
        for i in range(10):
            assert tensor[i].numpy().shape == sample.shape  # type: ignore
