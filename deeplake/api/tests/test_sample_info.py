from PIL import Image  # type: ignore
from PIL.ExifTags import TAGS  # type: ignore
from deeplake.util.exif import getexif
import numpy as np
import pytest
import os
import sys
import deeplake
import json


def get_exif_helper(path):
    img = Image.open(path)
    return json.loads(json.dumps(getexif(img)))


@pytest.mark.slow
def test_image_samples(local_ds_generator, compressed_image_paths):
    from PIL import ImageFile  # type: ignore

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    ds = local_ds_generator()
    jpg = ds.create_tensor("jpg_images", htype="image", sample_compression="jpg")
    jpg_paths = compressed_image_paths["jpeg"]
    for jpg_path in jpg_paths:
        ds.jpg_images.append(deeplake.read(jpg_path))

    for i, (jpg_path, sample_info) in enumerate(zip(jpg_paths, jpg.sample_info)):
        img = Image.open(jpg_path)
        sample_info2 = jpg[i].sample_info
        assert sample_info == sample_info2
        assert sample_info["exif"] == get_exif_helper(jpg_path)
        assert sample_info["shape"] == list(np.array(img).shape)
        assert sample_info["format"] == "jpeg"
        assert sample_info["filename"] == jpg_path

    ds = local_ds_generator()
    jpg = ds["jpg_images"]
    for i, (jpg_path, sample_info) in enumerate(zip(jpg_paths, jpg.sample_info)):
        img = Image.open(jpg_path)
        sample_info2 = jpg[i].sample_info
        assert sample_info == sample_info2
        assert sample_info["exif"] == get_exif_helper(jpg_path)
        assert sample_info["shape"] == list(np.array(img).shape)
        assert sample_info["format"] == "jpeg"
        assert sample_info["filename"] == jpg_path


@pytest.mark.skipif(
    os.name == "nt" and sys.version_info < (3, 7), reason="requires python 3.7 or above"
)
def test_video_samples(local_ds_generator, video_paths):
    import av  # type: ignore

    ds = local_ds_generator()
    mp4 = ds.create_tensor("mp4_videos", htype="video", sample_compression="mp4")
    mp4_paths = video_paths["mp4"]
    for mp4_path in mp4_paths:
        ds.mp4_videos.append(deeplake.read(mp4_path))

    for i, (mp4_path, sample_info) in enumerate(zip(mp4_paths, mp4.sample_info)):
        container = av.open(mp4_path)
        vstream = container.streams.video[0]

        sample_info2 = mp4[i].sample_info
        assert sample_info == sample_info2
        assert (
            sample_info["fps"]
            == vstream.guessed_rate.numerator / vstream.guessed_rate.denominator
        )
        assert sample_info["duration"] == vstream.duration or container.duration
        assert sample_info["format"] == "mp4"
        assert sample_info["filename"] == mp4_path

    ds = local_ds_generator()
    mp4 = ds["mp4_videos"]
    for i, (mp4_path, sample_info) in enumerate(zip(mp4_paths, mp4.sample_info)):
        container = av.open(mp4_path)
        vstream = container.streams.video[0]

        sample_info2 = mp4[i].sample_info
        assert sample_info == sample_info2
        assert (
            sample_info["fps"]
            == vstream.guessed_rate.numerator / vstream.guessed_rate.denominator
        )
        assert sample_info["duration"] == vstream.duration or container.duration
        assert sample_info["format"] == "mp4"
        assert sample_info["filename"] == mp4_path


@pytest.mark.skipif(
    os.name == "nt" and sys.version_info < (3, 7), reason="requires python 3.7 or above"
)
def test_audio_samples(local_ds_generator, audio_paths):
    import av  # type: ignore

    ds = local_ds_generator()
    mp3 = ds.create_tensor("mp3_audios", htype="audio", sample_compression="mp3")
    mp3_paths = [audio_paths["mp3"]]
    for mp3_path in mp3_paths:
        ds.mp3_audios.append(deeplake.read(mp3_path))

    for i, (mp3_path, sample_info) in enumerate(zip(mp3_paths, mp3.sample_info)):
        container = av.open(mp3_path)
        astream = container.streams.audio[0]

        sample_info2 = mp3[i].sample_info
        assert sample_info == sample_info2
        assert sample_info["nchannels"] == astream.channels
        assert sample_info["sample_rate"] == astream.sample_rate
        assert sample_info["duration"] == astream.duration or container.duration

    ds = local_ds_generator()
    mp3 = ds["mp3_audios"]
    for i, (mp3_path, sample_info) in enumerate(zip(mp3_paths, mp3.sample_info)):
        container = av.open(mp3_path)
        astream = container.streams.audio[0]

        sample_info2 = mp3[i].sample_info
        assert sample_info == sample_info2
        assert sample_info["nchannels"] == astream.channels
        assert sample_info["sample_rate"] == astream.sample_rate
        assert sample_info["duration"] == astream.duration or container.duration
