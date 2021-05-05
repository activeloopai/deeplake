import numpy as np
from typing import Generator, Union

from hub.core.chunk_engine.exceptions import ChunkGeneratorError


def generate_chunks(
    content_bytes: bytes,
    chunk_size: int,
    last_chunk_num_bytes: Union[None, int] = None,
) -> Generator[bytearray, None, None]:
    """
    Generator function that chunks bytes.

    Chunking is the process of taking the input `content_bytes` & breaking it up into a sequence of smaller bytes called "chunks".
    The sizes of each chunk are <= `chunk_size`.

    Example:
        content_bytes = b"1234567890123"
        chunk_size = 4
        yields:
            b"1234", 1
            b"5678", 2
            b"9012", 3
            b"3", 4

    Args:
        content_bytes (bytes): Bytes object with the data to be chunked.
        chunk_size (int): Each individual chunk will be assigned this many bytes maximum.
        last_chunk_bytes (int, optional): If chunks were created already, `last_chunk_bytes`
            should be set to the length of the last chunk created. This is so the generator's
            first output will be enough bytes to fill that chunk up to `chunk_size`.

    Yields:
        bytearray: Chunk of the `content_bytes`. Will have length on the interval (0, `chunk_size`].

    Raises:
        ChunkGeneratorError: If the provided `chunk_size` is smaller than the amount of bytes in the last chunk.
    """

    # validate inputs
    if chunk_size <= 0:
        raise ChunkGeneratorError("Cannot generate chunks of size <= 0.")
    if len(content_bytes) <= 0:
        return
    if last_chunk_num_bytes is None:
        bytes_left_in_last_chunk = 0
    else:
        if chunk_size < last_chunk_num_bytes:
            raise ChunkGeneratorError(
                "The provided `chunk_size` should be >= the number of bytes in the last chunk (%i < %i)."
                % (chunk_size, last_chunk_num_bytes)
            )

        bytes_left_in_last_chunk = chunk_size - last_chunk_num_bytes

    # yield the remainder of the last chunk (provided as `last_chunk_num_bytes`)
    total_bytes_yielded = 0
    if bytes_left_in_last_chunk > 0:
        chunk = content_bytes[:bytes_left_in_last_chunk]
        yield bytearray(chunk)
        total_bytes_yielded += bytes_left_in_last_chunk

    # yield all new chunks
    while total_bytes_yielded < len(content_bytes):
        end = total_bytes_yielded + chunk_size
        chunk = content_bytes[total_bytes_yielded:end]

        yield bytearray(chunk)
        total_bytes_yielded += len(chunk)
