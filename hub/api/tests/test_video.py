import os
import sys
import pytest

import hub
from hub.core.dataset import Dataset
from hub.util.exceptions import DynamicTensorNumpyError

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
    ("vstream_path", "hub_token"),
    [
        ("gcs_vstream_path", "hub_cloud_dev_token"),
        ("s3_vstream_path", "hub_cloud_dev_token"),
        ("hub_cloud_vstream_path", "hub_cloud_dev_token"),
    ],
    indirect=True,
)
def test_video_streaming(vstream_path, hub_token):
    ds = hub.load(vstream_path, read_only=True, token=hub_token)

    # no streaming, downloads chunk
    assert ds.mp4_videos[0].shape == (400, 360, 640, 3)
    assert ds.mp4_videos[0].numpy().shape == (400, 360, 640, 3)
    assert ds.mp4_videos[1].numpy().shape == (120, 1080, 1920, 3)

    # streaming
    assert ds.large_video[0].shape == (21312, 546, 1280, 3)
    assert ds.large_video[0, 13500].numpy().shape == (546, 1280, 3)
    # will use cached url
    assert ds.large_video[0, 18000].numpy().shape == (546, 1280, 3)


@pytest.mark.skipif(
    os.name == "nt" and sys.version_info < (3, 7), reason="requires python 3.7 or above"
)
@pytest.mark.parametrize(
    ("vstream_path", "hub_token"),
    [
        ("gcs_vstream_path", "hub_cloud_dev_token"),
        ("s3_vstream_path", "hub_cloud_dev_token"),
        ("hub_cloud_vstream_path", "hub_cloud_dev_token"),
    ],
    indirect=True,
)
def test_video_timestamps(vstream_path, hub_token):
    ds = hub.load(vstream_path, read_only=True, token=hub_token)

    with pytest.raises(ValueError):
        stamps = ds.mp4_videos[:2].timestamps

    stamps = ds.large_video[0, 12000:1199:-100].timestamps

    assert len(stamps) == 109

    # timestamp is 50, 24 fps video, 50 * 24 = 1200th frame
    assert stamps[-1] == 50

    # cover stepping without seeking
    stamps = ds.large_video[0, 1200:1300:2].timestamps

    assert len(stamps) == 50
    assert stamps[0] == 50


def test_video_exception(local_ds):
    with local_ds as ds:
        ds.create_tensor("abc")
        with pytest.raises(Exception):
            stamps = ds.abc.timestamps


@pytest.mark.skipif(
    os.name == "nt" and sys.version_info < (3, 7), reason="requires python 3.7 or above"
)
def test_video_sequence(local_ds, video_paths):
    with local_ds as ds:
        ds.create_tensor("video_seq", htype="sequence[video]", sample_compression="mp4")
        ds.video_seq.append([hub.read(video_paths["mp4"][0]) for _ in range(3)])
        ds.video_seq.append([hub.read(video_paths["mp4"][1]) for _ in range(3)])

        with pytest.raises(ValueError):
            ds.video_seq[:2].timestamps

        with pytest.raises(ValueError):
            ds.video_seq[0].timestamps

        with pytest.raises(ValueError):
            ds.video_seq[0, :2].timestamps

        assert ds.video_seq[0][1, 5:10].timestamps.shape == (5,)


@pytest.mark.skipif(
    os.name == "nt" and sys.version_info < (3, 7), reason="requires python 3.7 or above"
)
def test_video_data(local_ds, video_paths):
    with local_ds as ds:
        ds.create_tensor("video", htype="video", sample_compression="mp4")
        for _ in range(3):
            ds.video.append(hub.read(video_paths["mp4"][0]))
        ds.video.append(hub.read(video_paths["mp4"][1]))

        data = ds.video[2].data()
        assert data["frames"].shape == (400, 360, 640, 3)
        assert data["timestamps"].shape == (400,)

        data = ds.video[:2, 4, :5, :5].data()
        assert data["frames"].shape == (2, 5, 5, 3)
        assert data["timestamps"].shape == (2, 1)

        data = ds.video[:2, 10:20].data()
        assert data["frames"].shape == (2, 10, 360, 640, 3)
        assert data["timestamps"].shape == (2, 10)

        with pytest.raises(DynamicTensorNumpyError):
            ds.video[2:].data()

        data = ds.video[2:].data(aslist=True)
        assert len(data["frames"]) == 2
        assert data["frames"][0].shape == ds.video[2].shape
        assert data["frames"][1].shape == ds.video[3].shape
        assert len(data["timestamps"]) == 2
        assert data["timestamps"][0].shape == (ds.video[2].shape[0],)
        assert data["timestamps"][1].shape == (ds.video[3].shape[0],)


@pytest.mark.skipif(
    os.name == "nt" and sys.version_info < (3, 7), reason="requires python 3.7 or above"
)
def test_linked_video_timestamps(local_ds):
    with local_ds as ds:
        ds.create_tensor("videos", htype="link[video]")
        ds.videos.append(
            hub.link(
                "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4",
                creds_key="ENV",
            )
        )
        ds.videos[0, 5:10].timestamps == np.array(
            [0.04170833, 0.08341666, 0.125125, 0.16683333, 0.20854166]
        )
