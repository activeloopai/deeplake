import pytest
import itertools

import numpy as np

from hub.core.chunk_engine.generator import generate_chunks

from typing import Iterable


# configuration for tests. these configurations will be exhausted in a gridsearch.
# so the number of configs (or pytest.mark.parametrize sets) for each test will be
# the product of the lengths of the tuples used.
CHUNK_SIZES = (1, 99, 4096, 16000000)
BYTES_BATCH_SIZES = (1, 10)
PARTIAL_PERCENTAGES = (0.1, 0.3, 0.5, 0.7, 0.9)


def get_grid_search_params(*param_arrays: Iterable):
    """Gets all combinations of `param_arrays` (like a zip function, but instead of 1:1 correspondence, it's all combinations)."""
    return list(itertools.product(*param_arrays))


def get_dummy_bytes_batch(lengths: Iterable[int]) -> Iterable[bytes]:
    """Generate a list of random bytes with the provided lengths."""
    return [np.ones(length, dtype=bool).tobytes() for length in lengths]


def run_test(bytes_batch: Iterable[bytes], chunk_size: int):
    """
    General function to validate the chunk generator for the `bytes_batch` provided.

    Args:
        bytes_batch (Iterable[bytes): Iterable of bytes objects. These simulate incoming data
            to be chunked.
        chunk_size (int): The max size of the output chunks.
    """

    last_chunk_num_bytes = None
    for input_bytes in bytes_batch:
        relative_chunk_indices = []

        for chunk, relative_chunk_index in generate_chunks(
            input_bytes, chunk_size, last_chunk_num_bytes=last_chunk_num_bytes
        ):
            assert (
                len(chunk) == chunk_size
            ), "Each chunk is expected to have the same chunk_size."
            relative_chunk_indices.append(relative_chunk_index)

        assert len(relative_chunk_indices) == 1, "Only 1 chunk was expected."
        assert (
            relative_chunk_indices[0] == 1
        ), "Relative index should always indicate a new chunk with distance 1."

        last_chunk_num_bytes = len(chunk)


@pytest.mark.parametrize(
    "byte_batch_size,chunk_size", get_grid_search_params(BYTES_BATCH_SIZES, CHUNK_SIZES)
)
def test_perfect_fit(byte_batch_size: int, chunk_size: int):
    """
    Case:
        All bytes fit perfectly into chunks.

    Example:
        content_bytes (batch_size=1) = b"1094jfnv841q"
        chunk_size = 4
        output_chunks = [b"1094", b"jfnv", b"841q"]

        This is considered a "perfect fit" because all bytes fit perfectly into 3 chunks.
    """

    # create a batch of bytes that are all of length `chunk_size` (perfect fit)
    bytes_lengths = [chunk_size] * byte_batch_size
    bytes_batch = get_dummy_bytes_batch(bytes_lengths)

    run_test(bytes_batch, chunk_size)


# @pytest.mark.parametrize(
#     "bytes_lengths,chunk_size,partial_percentage",
#     get_grid_search_params(BYTES_BATCH_SIZES, CHUNK_SIZES, PARTIAL_PERCENTAGES),
# )
# def test_partial_fit(
#     bytes_lengths: Iterable[int], chunk_size: int, partial_percentage: float
# ):
#     """
#     Case:
#         Bytes don't perfectly fit into chunks.
#
#     Example:
#         content_bytes (batch_size=1) = b"f8bkmc99911c94"
#         chunk_size = 4
#         output_chunks = [b"f8bk", b"mc99", b"911c", b"94"]
#
#         The last chunk in `output_chunks` only has 2 bytes (chunk_size=4), which means
#         this is not a perfect fit, but rather it has a partial chunk at the end.
#
#         This partial chunk at the end needs to be filled by the chunks created by the
#         following bytes stream.
#     """
#
#     # TODO
#
#     bytes_batch = get_dummy_bytes_batch(bytes_lengths)
#     run_test(bytes_batch, chunk_size)


"""
    Example 3 (Fill Partial First):
        last_chunks = [b"f8bk", b"mc99", b"911c", b"94"]

        content_bytes = b"1c8494901048dcx"
        chunk_size = 4
        last_chunk_bytes = 2  # len(last_chunks[-1])
        output_chunks = [b"1c", b"8494", b"9010", b"48dc", b"x"]

        This example starts with `last_chunks` defined in the previous example (Example 2).
        The first output chunk is of length 2 because the last chunk in `last_chunks` can hold
        2 more bytes (`chunk_size - last_chunk_bytes == 2`). So the first yielded chunk is b"1c"
        to fill it.
"""

"""
# generate test parameters for `test_partial_end`
partial_end_params = []
for chunk_size in CHUNK_SIZES:
    for bytes_batch_size in BYTES_BATCH_SIZES:
        for partial_percentage in PARTIAL_PERCENTAGES:
            bytes_lengths = []
            partial_end_params.append((bytes_lengths, chunk_size))


@pytest.mark.parametrize("bytes_lengths,chunk_size", partial_end_params)
def test_partial_end(bytes_lengths: Iterable[int], chunk_size: int):
    run_test(bytes_lengths, chunk_size)
"""
