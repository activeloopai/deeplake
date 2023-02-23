import deeplake
from urllib.request import urlopen
from deeplake.util.keys import get_chunk_key
from deeplake.visualizer.video_streaming import _VideoStream
import pytest
import os
import sys


@pytest.mark.skipif(
    os.name == "nt" and sys.version_info < (3, 7), reason="requires python 3.7 or above"
)
def test_video_playback(local_ds_generator, video_paths):
    mp4_path = video_paths["mp4"][0]
    ds = local_ds_generator()
    ds.create_tensor("videos", htype="video", sample_compression="mp4")
    ds.videos.append(deeplake.read(mp4_path))
    enc = ds.videos.chunk_engine.chunk_id_encoder
    chunk_name = enc.get_name_for_chunk(0)
    chunk_key = get_chunk_key("videos", chunk_name, ds.version_state["commit_id"])
    stream = _VideoStream(ds.storage.next_storage, chunk_key)
    byte_stream = stream.read(0, 0, 10**6)[0]

    with open(mp4_path, "rb") as f:
        expected = f.read()

    assert byte_stream == expected


@pytest.mark.skipif(
    os.name == "nt" and sys.version_info < (3, 7), reason="requires python 3.7 or above"
)
def test_linked_video_playback(local_ds_generator, gcs_path):
    with local_ds_generator() as ds:
        ds.create_tensor("video_links", htype="link[video]", sample_compression="mp4")
        ds.add_creds_key("ENV")
        ds.populate_creds("ENV", from_environment=True)
        ds.video_links.append(
            deeplake.link(
                "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerJoyrides.mp4",
                creds_key="ENV",
            )
        )
        ds.video_links.append(
            deeplake.link(
                "gcs://gtv-videos-bucket/sample/ForBiggerJoyrides.mp4", creds_key="ENV"
            )
        )
        url = ds.video_links[0]._get_video_stream_url()
        assert (
            url
            == "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerJoyrides.mp4"
        )
        http_data = urlopen(url).read()

        url = ds.video_links[1]._get_video_stream_url()
        gcs_data = urlopen(url).read()

        assert gcs_data == http_data
