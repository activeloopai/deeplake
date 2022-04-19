import hub
from hub.util.keys import get_chunk_key
from hub.visualizer.video_streaming import _VideoStream


def test_video_playback(local_ds_generator, video_paths):
    mp4_path = video_paths["mp4"][0]
    ds = local_ds_generator()
    ds.create_tensor("videos", htype="video", sample_compression="mp4")
    ds.videos.append(hub.read(mp4_path))
    enc = ds.videos.chunk_engine.chunk_id_encoder
    chunk_name = enc.get_name_for_chunk(0)
    chunk_key = get_chunk_key("videos", chunk_name, ds.version_state["commit_id"])
    stream = _VideoStream(ds.storage.next_storage, chunk_key)
    byte_stream = stream.read(0, 0, 10**6)[0]

    with open(mp4_path, "rb") as f:
        expected = f.read()

    assert byte_stream == expected
