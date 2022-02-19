import os
import sys
import pytest

import hub
from hub.core.dataset import Dataset

import numpy as np


@pytest.mark.skipif(
    os.name == "nt" and sys.version_info < (3, 7), reason="requires python 3.7 or above"
)
@pytest.mark.parametrize("compression", hub.compression.VIDEO_COMPRESSIONS)
def test_video(local_ds, compression, video_paths):
    for i, path in enumerate(video_paths[compression]):
        tensor = local_ds.create_tensor(
            f"video_{i}", htype="video", sample_compression=compression
        )
        sample = hub.read(path)
        assert len(sample.shape) == 4
        if "dummy_data" in path:  # check shape only for internal test videos
            if compression == "mp4":
                assert sample.shape == (400, 360, 640, 3)
            elif compression == "mkv":
                assert sample.shape == (399, 360, 640, 3)
            elif compression == "avi":
                assert sample.shape == (901, 270, 480, 3)
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
def test_video_slicing(local_ds: Dataset, video_paths):
    for path in video_paths["mp4"]:
        if "samplemp4_1MB" in path:
            dummy = np.zeros((132, 720, 1080, 3))

            local_ds.create_tensor("video", htype="video", sample_compression="mp4")
            local_ds.video.append(hub.read(path))
            local_ds.video[0][0:5].numpy().shape == dummy[0:5].shape
            local_ds.video[0][100:120].numpy().shape == dummy[100:120].shape
            local_ds.video[0][120].numpy().shape == dummy[120].shape
            local_ds.video[0][10:5:-2].numpy().shape == dummy[10:5:-2].shape
            local_ds.video[0][-3:-10:-1].numpy().shape == dummy[-3:-10:-1].shape
            local_ds.video[0][-25:100:-2].numpy().shape == dummy[-25:100:-2].shape
            local_ds.video[0][::-1].numpy().shape == dummy[::-1].shape
            local_ds.video[0][:5:-1].numpy().shape == dummy[:5:-1].shape
            local_ds.video[0][-1].numpy().shape == dummy[-1].shape
            return
    raise Exception  # test did not run


@pytest.mark.skipif(
    os.name == "nt" and sys.version_info < (3, 7), reason="requires python 3.7 or above"
)
@pytest.mark.parametrize(
    "vstream_path",
    ["gcs_vstream_path", "s3_vstream_path", "hub_cloud_vstream_path"],
    indirect=True,
)
def test_video_streaming(vstream_path, hub_cloud_dev_token):
    if vstream_path.startswith("hub://"):
        ds = hub.load(vstream_path, read_only=True, token=hub_cloud_dev_token)
    else:
        ds = hub.load(vstream_path, read_only=True)

    # no streaming, downloads chunk
    assert ds.mp4_videos[0].shape == (400, 360, 640, 3)
    assert ds.mp4_videos[0].numpy().shape == (400, 360, 640, 3)
    assert ds.mp4_videos[1].numpy().shape == (120, 1080, 1920, 3)

    # streaming
    assert ds.large_video[0].shape == (21312, 546, 1280, 3)
    assert ds.large_video[0, 13500].numpy().shape == (546, 1280, 3)
    # will use cached url
    assert ds.large_video[0, 18000].numpy().shape == (546, 1280, 3)
