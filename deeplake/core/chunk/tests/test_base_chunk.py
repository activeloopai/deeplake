import pytest

from deeplake.core.chunk.uncompressed_chunk import UncompressedChunk
from deeplake.core.meta import TensorMeta


def test_text_sample_to_byte_string():
    chunk = UncompressedChunk(
        min_chunk_size=10,
        max_chunk_size=1000,
        tiling_threshold=1000,
        tensor_meta=TensorMeta(),
    )

    assert chunk._text_sample_to_byte_string("test") == b"test"
    assert chunk._text_sample_to_byte_string(3) == b"3"
    assert chunk._text_sample_to_byte_string(3.5) == b"3.5"
    assert chunk._text_sample_to_byte_string(None) == b""
    assert chunk._text_sample_to_byte_string([]) == b""
