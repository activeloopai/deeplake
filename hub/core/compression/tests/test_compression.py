from typing import Union

import numpy as np
import pytest

from hub.core.compression import BaseNumCodec, BaseImgCodec
from hub.core.compression.jpeg import JPEG
from hub.core.compression.lz4 import LZ4
from hub.core.compression.numpy import NUMPY
from hub.core.compression.png import PNG
from hub.core.compression.webp import WEBP
from hub.core.compression.zstd import ZSTD

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
SHAPE_PARAM = "shape"
FROM_CONFIG_PARAM = "from_config"
COMPRESSOR_PARAM = "compressor"
SINGLE_CHANNEL_PARAM = "single_channel"
DTYPE_PARAM = "dtype"
ACCELERATION_PARAM = "acceleration"
LEVEL_PARAM = "level"
parametrize_image_shape = pytest.mark.parametrize(SHAPE_PARAM, IMG_ARRAY_SHAPES)
parametrize_generic_array_shape = pytest.mark.parametrize(
    SHAPE_PARAM, GENERIC_ARRAY_SHAPES
)
parametrize_benchmark_shape = pytest.mark.parametrize(SHAPE_PARAM, BENCHMARK_SHAPES)
paramentrize_dtypes = pytest.mark.parametrize(DTYPE_PARAM, ARRAY_DTYPES)
parametrize_lz4_accelerations = pytest.mark.parametrize(
    ACCELERATION_PARAM, LZ4_ACCELERATIONS
)
parametrize_zstd_levels = pytest.mark.parametrize(LEVEL_PARAM, ZSTD_LEVELS)
parametrize_compressor = pytest.mark.parametrize(
    COMPRESSOR_PARAM, IMG_CODECS + NUM_CODECS
)
parametrize_single_channel = pytest.mark.parametrize(
    SINGLE_CHANNEL_PARAM, (False, True)
)
parametrize_from_config = pytest.mark.parametrize(FROM_CONFIG_PARAM, (False, True))


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


@parametrize_image_shape
@parametrize_from_config
def test_png_codec(from_config: bool, shape: tuple) -> None:
    codec = PNG()
    if from_config:
        config = codec.get_config()
        codec = codec.from_config(config)
    arr = np.ones(shape, dtype="uint8")
    check_equals_decoded(arr, codec)


@parametrize_single_channel
def test_png_codec_config(single_channel: bool) -> None:
    codec = PNG(single_channel=single_channel)
    check_codec_config(codec)


@parametrize_single_channel
def test_png_codec_single_channel(single_channel: bool) -> None:
    codec = PNG(single_channel=single_channel)
    check_codec_single_channel(codec)


@parametrize_from_config
@parametrize_image_shape
def test_jpeg_codec(from_config: bool, shape: tuple) -> None:
    codec = JPEG()
    if from_config:
        config = codec.get_config()
        codec = JPEG.from_config(config)
    arr = np.ones(shape, dtype="uint8")
    check_equals_decoded(arr, codec)


@parametrize_single_channel
def test_jpeg_codec_config(single_channel: bool) -> None:
    codec = JPEG(single_channel=single_channel)
    check_codec_config(codec)


@parametrize_single_channel
def test_jpeg_codec_single_channel(single_channel: bool) -> None:
    codec = JPEG(single_channel=single_channel)
    check_codec_single_channel(codec)


@parametrize_from_config
@parametrize_image_shape
def test_webp_codec(from_config: bool, shape: tuple) -> None:
    codec = WEBP()
    if from_config:
        config = codec.get_config()
        codec = WEBP.from_config(config)
    arr = np.ones(shape, dtype="uint8")
    check_equals_decoded(arr, codec)


@parametrize_single_channel
def test_webp_codec_config(single_channel: bool) -> None:
    codec = WEBP(single_channel=single_channel)
    check_codec_config(codec)


@parametrize_single_channel
def test_webp_codec_single_channel(single_channel: bool) -> None:
    codec = WEBP(single_channel=single_channel)
    check_codec_single_channel(codec)


@parametrize_lz4_accelerations
@parametrize_generic_array_shape
def test_lz4(acceleration: int, shape: tuple) -> None:
    codec = LZ4(acceleration=acceleration)
    arr = np.random.rand(*shape)
    check_equals_decoded(arr, codec)


@parametrize_generic_array_shape
def test_numpy(shape: tuple) -> None:
    codec = NUMPY()
    arr = np.random.rand(*shape)
    check_equals_decoded(arr, codec)


@parametrize_zstd_levels
@parametrize_generic_array_shape
def test_zstd(level: int, shape: tuple) -> None:
    codec = ZSTD(level=level)
    arr = np.random.rand(*shape)
    check_equals_decoded(arr, codec)


def test_base_name():
    class RandomCompressor(BaseImgCodec):
        def __init__(single_channel=True):
            super().__init__()

    new_compressor = RandomCompressor()
    assert new_compressor.__name__ == "randomcompressor"


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


@pytest.mark.full_benchmark
@pytest.mark.benchmark(group="compressor_encode")
@parametrize_benchmark_shape
@parametrize_compressor
def test_encode_speed(benchmark, compressor, shape):
    compressor, comp_input = get_compr_input(compressor, shape)
    benchmark(compressor.encode, comp_input)


@pytest.mark.full_benchmark
@pytest.mark.benchmark(group="compressor_decode")
@parametrize_benchmark_shape
@parametrize_compressor
def test_decode_speed(benchmark, compressor, shape):
    compressor, comp_input = get_compr_input(compressor, shape)
    bytes = compressor.encode(comp_input)
    benchmark(compressor.decode, bytes)
