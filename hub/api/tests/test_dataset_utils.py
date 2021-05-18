import pytest
from hub.api.dataset_utils import get_compressor
from hub.util.exceptions import InvalidCompressor


@pytest.mark.parametrize("codec_name", (mod for mod in dir() if mod.isupper()))
def test_get_compressor(codec_name: str) -> None:
    compressor = get_compressor(codec_name)
    assert compressor.__name__ == codec_name


def test_unavailable_codec():
    codec_name = "zip"
    with pytest.raises(InvalidCompressor):
        get_compressor(codec_name)
