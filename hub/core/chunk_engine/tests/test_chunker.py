from typing import List, Tuple

import pytest

from hub.core.chunk_engine.chunker import generate_chunks

# chunk_size,bytes_batch,expected_chunks
PERFECT_FIT: Tuple = (
    (1, [b"1"], [b"1"]),
    (1, [b"1", b"2"], [b"1", b"2"]),
    (1, [b"1", b"2", b"3"], [b"1", b"2", b"3"]),
    (1, [b"1234"], [b"1", b"2", b"3", b"4"]),
    (4, [b"1234"], [b"1234"]),
    (4, [b"1234", b"5678"], [b"1234", b"5678"]),
    (4, [b"12345678"], [b"1234", b"5678"]),
    (10, [b"12", b"3456", b"78", b"9", b"0"], [b"1234567890"]),
)

# chunk_size,bytes_batch,expected_chunks
PARTIAL_FIT: Tuple = (
    (1, [b""], []),
    (2, [b"1"], [b"1"]),
    (2, [b"123"], [b"12", b"3"]),
    (4, [b"123"], [b"123"]),
    (4, [b"1234567"], [b"1234", b"567"]),
    (4, [b"1234567"], [b"1234", b"567"]),
    (8, [b"1", b"2", b"3", b"4", b"5", b"6", b"7"], [b"1234567"]),
)


@pytest.mark.parametrize("chunk_size,bytes_batch,expected_chunks", PERFECT_FIT)
def test_generate_perfect_fit(
    chunk_size: int, bytes_batch: List[bytes], expected_chunks: List[bytes]
):
    """All output chunks should be equal in length to `chunk_size`."""
    run_generate_chunks_test(chunk_size, bytes_batch, expected_chunks)


@pytest.mark.parametrize("chunk_size,bytes_batch,expected_chunks", PARTIAL_FIT)
def test_generate_partial_fit(
    chunk_size: int, bytes_batch: List[bytes], expected_chunks: List[bytes]
):
    """Output chunks may differ in length to `chunk_size`."""
    run_generate_chunks_test(chunk_size, bytes_batch, expected_chunks)


def run_generate_chunks_test(
    chunk_size: int, bytes_batch: List[bytes], expected_chunks: List[bytes]
):
    """
    This method iterates through the `chunk_generator(...)` & keeps a running list of the chunks.

    When a chunk is yielded, it either adds it to the end of the previous chunk yielded (if it
        exists & is not full) or creates a new chunk.
    """

    actual_chunks: List[bytearray] = []
    bytes_left_in_last_chunk: int = 0
    for bytes_object in bytes_batch:
        for chunk_bytes in generate_chunks(
            memoryview(bytes_object),
            chunk_size,
            bytes_left_in_last_chunk=bytes_left_in_last_chunk,
        ):
            chunk = bytearray(chunk_bytes)

            # fill last chunk if possible, otherwise create new chunk
            if len(actual_chunks) <= 0 or len(actual_chunks[-1]) >= chunk_size:
                actual_chunks.append(chunk)
            else:
                actual_chunks[-1].extend(chunk)

            bytes_left_in_last_chunk = chunk_size - len(chunk)

    assert actual_chunks == expected_chunks
