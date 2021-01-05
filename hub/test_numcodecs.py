import numpy as np

from .numcodecs import PngCodec


def test_png_codec():
    codec = PngCodec()
    arr = np.ones((1000, 1000, 3), dtype="uint8")
    bytes_ = codec.encode(arr)
    arr_ = codec.decode(bytes_)
    assert (arr == arr_).all()
