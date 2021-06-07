from typing import Generator

from hub.util.exceptions import ChunkSizeTooSmallError


def generate_chunks(
    content_bytes: memoryview,
    chunk_size: int,
    bytes_left_in_last_chunk: int = 0,
) -> Generator[memoryview, None, None]:
    """Generator function that chunks bytes (as memoryview).

    Chunking is the process of taking the input `content_bytes` and breaking it up into a sequence of smaller bytes
    called "chunks".
    The sizes of each chunk are <= `chunk_size`.

    Example:
        content_bytes = b"1234567890123"
        chunk_size = 4
        yields:
            b"1234"
            b"5678"
            b"9012"
            b"3"

    Args:
        content_bytes (memoryview): Memoryview of bytes to be chunked.
        chunk_size (int): Each individual chunk will be assigned this many bytes maximum.
        bytes_left_in_last_chunk (int): If chunks were created already, `bytes_left_in_last_chunk`
            should be set to the `chunk_size - len(last_chunk)`. This is so the generator's
            first output will be enough bytes to fill that chunk up to `chunk_size`.

    Yields:
        bytes: Chunk of the `content_bytes`. Will have length on the interval (1, `chunk_size`].

    Raises:
        ChunkSizeTooSmallError: If `chunk_size` <= 0
        ValueError: If `bytes_left_in_last_chunk` < 0
    """

    # validate inputs
    if chunk_size <= 0:
        raise ChunkSizeTooSmallError()
    if bytes_left_in_last_chunk < 0:
        raise ValueError("Bytes left in last chunk must be >= 0.")
    if len(content_bytes) <= 0:
        return

    # yield the remainder of the last chunk (provided as `last_chunk_num_bytes`)
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
