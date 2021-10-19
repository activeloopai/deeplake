import pytest
from hub.tests.dataset_fixtures import enabled_datasets

import hub
from hub.core.dataset import Dataset


@enabled_datasets
@pytest.mark.parametrize("compression", hub.compression.VIDEO_COMPRESSIONS)
def test_video(ds: Dataset, compression, video_paths):
    path = video_paths[compression]
    ds.create_tensor("video", htype="video", sample_compression=compression)
    sample = hub.read(path)
    assert len(sample.shape) == 4
    if compression in ("mp4", "mkv"):
        assert sample.shape == (400, 360, 640, 3)
    elif compression == "avi":
        assert sample.shape == (900, 270, 480, 3)
    assert sample.shape[-1] == 3
    with ds:
        for _ in range(5):
            ds.video.append(hub.read(path))  # type: ignore
        ds.video.extend([hub.read(path) for _ in range(5)])  # type: ignore
    for i in range(10):
        assert ds.video[i].numpy().shape == sample.shape  # type: ignore
