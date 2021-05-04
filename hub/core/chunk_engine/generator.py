import numpy as np
from typing import Generator, Tuple

from hub.core.chunk_engine.exceptions import ChunkGeneratorError


def generate_chunks(
    content_bytes: bytes,
    chunk_size: int,
    last_chunk_num_bytes: int = None,
) -> Generator[Tuple[bytes, int], None, None]:
    """
    Generator function that chunks bytes.

    Chunking is the process of taking the input `content_bytes` & breaking it up into a sequence of smaller bytes called "chunks".
    The sizes of each chunk are <= `chunk_size`.

    Example:
        content_bytes = b"1094jfnv841qx"
        chunk_size = 4
        yields:
            b"1094", 1
            b"jfnv", 2
            b"841q", 3
            b"x", 4

    Args:
        content_bytes (bytes): Bytes object with the data to be chunked.
        chunk_size (int): Each individual chunk will be assigned this many bytes maximum.
        last_chunk_bytes (int, optional): If chunks were created already, `last_chunk_bytes`
            should be set to the length of the last chunk created. This is so the generator's
            first output will be enough bytes to fill that chunk up to `chunk_size`.

    Yields:
        bytes: Chunk of the `content_bytes`. Will have length on the interval (0, `chunk_size`].
        int: Relative index of the yielded chunk (bytes) to previously created chunks (if any).
            0: Append yielded chunk (bytes) to the previous chunk (bytes). 0 is only possible when
                `last_chunk_num_bytes` is provided and is less than `chunk_size`.
            1: Create the first new chunk using yielded chunk (bytes).
            2: Create the second new chunk using yielded chunk (bytes).
            3+: ...

    Raises:
        ChunkGeneratorError: If the provided `chunk_size` is smaller than the amount of bytes in the last chunk.
    """

    # validate inputs
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

    content_num_bytes = len(content_bytes)

    # yield the remainder of the last chunk (provided as `last_chunk_num_bytes`)
    total_bytes_yielded = 0
    if bytes_left_in_last_chunk > 0:
        content_bytes_piece = content_bytes[:bytes_left_in_last_chunk]
        yield content_bytes_piece, 0
        total_bytes_yielded += bytes_left_in_last_chunk

    num_chunks_to_create = max(
        1, int(np.ceil((content_num_bytes - total_bytes_yielded) / chunk_size))
    )
    start_chunk = 1

    # yield all chunks that are exactly equal to `chunk_size`
    for piece_index, relative_chunk_index in enumerate(
        range(start_chunk, num_chunks_to_create + start_chunk)
    ):
        end = total_bytes_yielded + chunk_size
        content_bytes_piece = content_bytes[total_bytes_yielded:end]

        if total_bytes_yielded >= content_num_bytes:
            # prevents empty pieces being generated
            break

        yield content_bytes_piece, relative_chunk_index
        total_bytes_yielded += len(content_bytes_piece)
