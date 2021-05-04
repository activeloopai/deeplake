import pytest

import numpy as np

from hub.core.chunk_engine.generator import generate_chunks

from typing import List


def get_dummy_bytes_list(lengths: List[int]) -> List[bytes]:
    """Generate a list of random bytes with the provided lengths."""
    return [np.ones(length, dtype=bool).tobytes() for length in lengths]


CHUNK_SIZES = (1, 99, 4096, 16000000)
BYTES_BATCH_SIZES = (1, 10)


# generate test parameters for `test_perfect_fit`
perfect_fit_params = []
for chunk_size in CHUNK_SIZES:
    for bytes_batch_size in BYTES_BATCH_SIZES:
        perfect_fit_params.append(([chunk_size] * bytes_batch_size, chunk_size))


@pytest.mark.parametrize("bytearray_list_lengths,chunk_size", perfect_fit_params)
def test_perfect_fit(bytearray_list_lengths: List[int], chunk_size: int):
    bytes_batch = get_dummy_bytes_list(bytearray_list_lengths)

    for input_bytes in bytes_batch:
        for chunk, relative_chunk_index in generate_chunks(input_bytes, chunk_size):
            assert (
                len(chunk) == chunk_size
            ), "Each chunk is expected to have the same chunk_size."
