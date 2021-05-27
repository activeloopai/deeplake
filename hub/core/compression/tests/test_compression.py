from typing import Union
import pytest
import numpy as np
from hub.core.compression.numpy import NUMPY
from hub.core.compression.zstd import ZSTD
from hub.core.compression.lz4 import LZ4
from hub.core.compression.webp import WEBP
from hub.core.compression.jpeg import JPEG
from hub.core.compression.png import PNG
from hub.core.compression import BaseNumCodec, BaseImgCodec

IMG_ARRAY_SHAPES = (
    (30, 30, 3),
    (2, 30, 30, 3),
    (30, 30, 1),
)
GENERIC_ARRAY_SHAPES = ((20, 1), (1, 20), ((60, 60)))
BENCHMARK_SHAPES = ((30, 30, 3), (5000, 5000, 3))
ARRAY_DTYPES = (
    "uint8",
    "uint16",
    "float16",
    "float32",
)
LZ4_ACCELERATIONS = (1, 50, 100)
ZSTD_LEVELS = (1, 11, 22)
IMG_CODECS = (PNG, JPEG, WEBP)
NUM_CODECS = (LZ4, ZSTD, NUMPY)


def check_equals_decoded(
    actual_array: np.ndarray, codec: Union[BaseImgCodec, BaseNumCodec]
) -> None:
    bytes_ = codec.encode(actual_array)
    if isinstance(codec, BaseNumCodec):
        bytes_double = codec.encode(bytes_)
        bytes_ = codec.decode(bytes_double)
    decoded_array = codec.decode(bytes_)
    assert (actual_array == decoded_array).all()


def check_codec_config(codec: Union[BaseImgCodec]) -> None:
    config = codec.get_config()
    assert config["id"] == codec.__name__
    assert config["single_channel"] == codec.single_channel


def check_codec_single_channel(codec: Union[BaseImgCodec]) -> None:
    if codec.single_channel:
        arr = np.ones((100, 100, 1), dtype="uint8")
    else:
        arr = np.ones((100, 100), dtype="uint8")
    check_equals_decoded(arr, codec)


@pytest.mark.parametrize("from_config", [False, True])
@pytest.mark.parametrize("shape", IMG_ARRAY_SHAPES)
def test_png_codec(from_config: bool, shape: tuple) -> None:
    codec = PNG()
    if from_config:
        config = codec.get_config()
        codec = codec.from_config(config)
    arr = np.ones(shape, dtype="uint8")
    check_equals_decoded(arr, codec)


@pytest.mark.parametrize("single_channel", [False, True])
def test_png_codec_config(single_channel: bool) -> None:
    codec = PNG(single_channel=single_channel)
    check_codec_config(codec)


@pytest.mark.parametrize("single_channel", [False, True])
def test_png_codec_single_channel(single_channel: bool) -> None:
    codec = PNG(single_channel=single_channel)
    check_codec_single_channel(codec)


@pytest.mark.parametrize("from_config", [False, True])
@pytest.mark.parametrize("shape", IMG_ARRAY_SHAPES)
def test_jpeg_codec(from_config: bool, shape: tuple) -> None:
    compr = JPEG()
    if from_config:
        config = compr.get_config()
        compr = JPEG.from_config(config)
    arr = np.ones(shape, dtype="uint8")
    check_equals_decoded(arr, compr)


@pytest.mark.parametrize("single_channel", [False, True])
def test_jpeg_codec_config(single_channel: bool) -> None:
    codec = JPEG(single_channel=single_channel)
    check_codec_config(codec)


@pytest.mark.parametrize("single_channel", [False, True])
def test_jpeg_codec_single_channel(single_channel: bool) -> None:
    codec = JPEG(single_channel=single_channel)
    check_codec_single_channel(codec)


@pytest.mark.parametrize("from_config", [False, True])
@pytest.mark.parametrize("shape", IMG_ARRAY_SHAPES)
def test_webp_codec(from_config: bool, shape: tuple) -> None:
    codec = WEBP()
    if from_config:
        config = codec.get_config()
        codec = WEBP.from_config(config)
    arr = np.ones(shape, dtype="uint8")
    check_equals_decoded(arr, codec)


@pytest.mark.parametrize("single_channel", [False, True])
def test_webp_codec_config(single_channel: bool) -> None:
    codec = WEBP(single_channel=single_channel)
    check_codec_config(codec)


@pytest.mark.parametrize("single_channel", [False, True])
def test_webp_codec_single_channel(single_channel: bool) -> None:
    codec = WEBP(single_channel=single_channel)
    check_codec_single_channel(codec)


@pytest.mark.parametrize("acceleration", LZ4_ACCELERATIONS)
@pytest.mark.parametrize("shape", GENERIC_ARRAY_SHAPES)
def test_lz4(acceleration: int, shape: tuple) -> None:
    codec = LZ4(acceleration=acceleration)
    arr = np.random.rand(*shape)
    check_equals_decoded(arr, codec)


@pytest.mark.parametrize("shape", GENERIC_ARRAY_SHAPES)
def test_numpy(shape: tuple) -> None:
    codec = NUMPY()
    arr = np.random.rand(*shape)
    check_equals_decoded(arr, codec)


@pytest.mark.parametrize("level", ZSTD_LEVELS)
@pytest.mark.parametrize("shape", GENERIC_ARRAY_SHAPES)
def test_zstd(level: int, shape: tuple) -> None:
    codec = ZSTD(level=level)
    arr = np.random.rand(*shape)
    check_equals_decoded(arr, codec)


def test_base_name():
    class RandomCompressor(BaseImgCodec):
        def __init__(single_channel=True):
            super().__init__()

    new_compressor = RandomCompressor()
    with pytest.raises(NotImplementedError):
        new_compressor.__name__


def test_kwargs():
    with pytest.raises(ValueError):
        compressor = LZ4(another_kwarg=1)
    with pytest.raises(ValueError):
        compressor = ZSTD(another_kwarg=1)
    with pytest.raises(ValueError):
        compressor = PNG(another_kwarg=1)
    with pytest.raises(ValueError):
        compressor = JPEG(another_kwarg=1)
    with pytest.raises(ValueError):
        compressor = WEBP(another_kwarg=1)


def get_compr_input(compressor, shape):
    comp_input = np.random.randint(0, 255, size=shape, dtype="uint8")
    if compressor in NUM_CODECS:
        comp_input = comp_input.tobytes()
    if compressor == WEBP:
        compressor = compressor(quality=85)
    else:
        compressor = compressor()
    return compressor, comp_input


@pytest.mark.parametrize("shape", BENCHMARK_SHAPES)
@pytest.mark.parametrize("compressor", IMG_CODECS + NUM_CODECS)
def test_encode_speed(benchmark, compressor, shape):
    compressor, comp_input = get_compr_input(compressor, shape)
    benchmark(compressor.encode, comp_input)


@pytest.mark.parametrize("shape", BENCHMARK_SHAPES)
@pytest.mark.parametrize("compressor", IMG_CODECS + NUM_CODECS)
def test_decode_speed(benchmark, compressor, shape):
    compressor, comp_input = get_compr_input(compressor, shape)
    bytes = compressor.encode(comp_input)
    benchmark(compressor.decode, bytes)
