import pytest
from hub.api.dataset_utils import _get_compressor

CODECS = ("lz4", "zstd", "numpy", "png", "jpeg")


@pytest.mark.parametrize("codec_name", CODECS)
def test_get_compressor(codec_name: str) -> None:
    compressor = _get_compressor(codec_name)
    assert compressor.__name__ == codec_name
    compressor = _get_compressor(None)
    assert compressor is None


def test_unavailable_codec():
    codec_name = "zip"
    with pytest.raises(ValueError):
        _get_compressor(codec_name)
