import pytest
import itertools

import numpy as np

from hub.core.chunk_engine.generator import generate_chunks

from typing import List, Union, Tuple


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
def test_perfect_fit(
    chunk_size: int, bytes_batch: List[bytes], expected_chunks: List[bytes]
):
    run_test(chunk_size, bytes_batch, expected_chunks)


@pytest.mark.parametrize("chunk_size,bytes_batch,expected_chunks", PARTIAL_FIT)
def test_partial_fit(
    chunk_size: int, bytes_batch: List[bytes], expected_chunks: List[bytes]
):
    run_test(chunk_size, bytes_batch, expected_chunks)


def run_test(chunk_size: int, bytes_batch: List[bytes], expected_chunks: List[bytes]):
    actual_chunks: List[bytearray] = []
    global_relative_indices: List[int] = []
    last_chunk_num_bytes: Union[int, None] = None
    chunk = None
    for bytes_object in bytes_batch:
        relative_indices = []
        for chunk, relative_index in generate_chunks(
            bytes_object,
            chunk_size,
            last_chunk_num_bytes=last_chunk_num_bytes,
        ):
            if relative_index == 0:
                actual_chunks[-1].extend(chunk)
            else:
                actual_chunks.append(chunk)

            relative_indices.append(relative_index)
            global_relative_indices.append(relative_index)

        assert np.all(
            np.diff(relative_indices) == 1
        ), "Relative chunk indices are expected to be a sequential counter."

        if chunk is not None:
            last_chunk_num_bytes = len(chunk)

    assert len(actual_chunks) == len(
        expected_chunks
    ), "Got the wrong amount of output chunks. Relative indices: %s" % (
        str(global_relative_indices)
    )
    for actual_chunk, expected_chunk in zip(actual_chunks, expected_chunks):
        assert actual_chunk == expected_chunk
