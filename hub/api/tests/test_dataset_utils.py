import pytest
from hub.api.dataset_utils import _get_compressor
import numcodecs
import numcodecs.lz4
import numcodecs.zstd
from hub.numcodecs import PngCodec


def test_get_compression():
    assert _get_compressor("lz4") == numcodecs.LZ4(numcodecs.lz4.DEFAULT_ACCELERATION)
    assert _get_compressor(None) is None
    assert _get_compressor("default") == "default"
    assert _get_compressor("zstd") == numcodecs.Zstd(numcodecs.zstd.DEFAULT_CLEVEL)
    assert _get_compressor("png") == PngCodec(solo_channel=True)
    with pytest.raises(ValueError):
        _get_compressor("abcd")
