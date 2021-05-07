import pytest
import numpy as np

from hub.codec import PngCodec, NumPy, Lz4, Zstd, JpegCodec


@pytest.mark.parametrize("from_config", [False, True])
def test_png_codec(from_config: bool) -> None:
    codec = PngCodec()
    if from_config:
        config = codec.get_config()
        codec = PngCodec.from_config(config)
    arr = np.ones((1000, 1000, 3), dtype="uint8")
    bytes_ = codec.encode(arr)
    arr_ = codec.decode(bytes_)
    assert (arr == arr_).all()


@pytest.mark.parametrize("single_channel", [False, True])
def test_png_codec_config(single_channel: bool) -> None:
    codec = PngCodec(single_channel)
    config = codec.get_config()
    assert config["id"] == "png"
    assert config["single_channel"] == single_channel


@pytest.mark.parametrize("single_channel", [False, True])
def test_png_codec_single_channel(single_channel: bool) -> None:
    codec = PngCodec(single_channel)
    if single_channel:
        arr = np.ones((1000, 1000, 1), dtype="uint8")
    else:
        arr = np.ones((1000, 1000), dtype="uint8")
    bytes_ = codec.encode(arr)
    arr_ = codec.decode(bytes_)
    assert (arr == arr_).all()


@pytest.mark.parametrize("from_config", [False, True])
def test_jpeg_codec(from_config: bool) -> None:
    codec = JpegCodec()
    if from_config:
        config = codec.get_config()
        codec = JpegCodec.from_config(config)
    arr = np.ones((1000, 1000, 3), dtype="uint8")
    bytes_ = codec.encode(arr)
    arr_ = codec.decode(bytes_)
    assert (arr == arr_).all()


@pytest.mark.parametrize("single_channel", [False, True])
def test_jpeg_codec_config(single_channel: bool) -> None:
    codec = JpegCodec(single_channel)
    config = codec.get_config()
    assert config["id"] == "jpeg"
    assert config["single_channel"] == single_channel


@pytest.mark.parametrize("single_channel", [False, True])
def test_jpeg_codec_single_channel(single_channel: bool) -> None:
    codec = JpegCodec(single_channel)
    if single_channel:
        arr = np.ones((1000, 1000, 1), dtype="uint8")
    else:
        arr = np.ones((1000, 1000), dtype="uint8")
    bytes_ = codec.encode(arr)
    arr_ = codec.decode(bytes_)
    assert (arr == arr_).all()


def test_lz4() -> None:
    codec = Lz4(acceleration=3)
    arr = np.random.rand(3, 100)
    bytes_ = codec.encode(arr)
    arr_ = codec.decode(bytes_)
    assert np.all(arr == arr_)


def test_numpy() -> None:
    codec = NumPy()
    arr = np.random.rand(3, 100)
    bytes_ = codec.encode(arr)
    arr_ = codec.decode(bytes_)
    assert np.all(arr == arr_)


def test_zstd() -> None:
    codec = Zstd(level=1)
    arr = np.random.rand(3, 100)
    bytes_ = codec.encode(arr)
    arr_ = codec.decode(bytes_)
    assert np.all(arr == arr_)
