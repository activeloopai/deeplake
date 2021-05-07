from typing import Generator, Optional

from hub.util.exceptions import ChunkSizeTooSmallError


def generate_chunks(
    content_bytes: bytes,
    chunk_size: int,
    last_chunk_num_bytes: Optional[int] = None,
) -> Generator[bytes, None, None]:
    """
    Generator function that chunks some `content_bytes` into a sequence of
    smaller bytes of size <= `chunk_size`

    Example:
        content_bytes = b"1234567890123"
        chunk_size = 4
        yields:
            b"1234"
            b"5678"
            b"9012"
            b"3"

    Args:
        content_bytes (bytes): Bytes object with the data to be chunked.
        chunk_size (int): Desired size of yielded chunks.
        last_chunk_num_bytes (int, optional): If chunks were created already,
            `last_chunk_num_bytes` should be given so that the first output
            can yield the remaining bytes to fill the last previous chunk.

    Yields:
        bytes: Chunk of the `content_bytes`. Has length <= `chunk_size`.

    Raises:
        ValueError: If `chunk_size` is <= 0
        ChunkSizeTooSmallError: If `chunk_size` < `last_chunk_num_bytes`
    """

    # validate inputs
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if len(content_bytes) <= 0:
        return
    if last_chunk_num_bytes is None:
        bytes_left_in_last_chunk = 0
    else:
        if chunk_size < last_chunk_num_bytes:
            raise ChunkSizeTooSmallError()

        bytes_left_in_last_chunk = chunk_size - last_chunk_num_bytes

    # yield the remainder of the last chunk (from `last_chunk_num_bytes`)
    total_bytes_yielded = 0
    if bytes_left_in_last_chunk > 0:
        chunk = content_bytes[:bytes_left_in_last_chunk]
        yield chunk
        total_bytes_yielded += bytes_left_in_last_chunk

    # yield all new chunks
    while total_bytes_yielded < len(content_bytes):
        end = total_bytes_yielded + chunk_size
        chunk = content_bytes[total_bytes_yielded:end]

        yield chunk
        total_bytes_yielded += len(chunk)
