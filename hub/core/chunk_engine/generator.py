import numpy as np
from typing import Generator, Tuple

from hub.core.chunk_engine.exceptions import ChunkGeneratorError


def chunk(
    content_bytes: bytes,
    chunk_size: int,
    last_chunk_num_bytes: int = None,
) -> Generator[Tuple[bytes, int], None, None]:
    """
    Generator function that chunks bytes.

    Chunking is the process of taking the input `content_bytes` & breaking it up into a sequence of smaller bytes called "chunks".
    The sizes of each chunk are <= `chunk_size`.

    Example 1:
        content_bytes = b"1094jfnv841q"
        chunk_size = 4
        output_chunks = [b"1094", b"jfnv", b"841q"]

        This is considered a "perfect fit" because all bytes fit perfectly into 3 chunks.


    Example 2:
        content_bytes = b"f8bkmc99911c94"
        chunk_size = 4
        output_chunks = [b"f8bk", b"mc99", b"911c", b"94"]

        As you can see, the last chunk in `output_chunks` only has 2 bytes (chunk_size=4), which means
        this is not a perfect fit, but rather it has a partial chunk at the end.


    Args:
        content_bytes (bytes): Bytes object with the data to be chunked.
        chunk_size (int): Each individual chunk will be assigned this many bytes maximum.
        last_chunk_bytes (int, optional): If chunks were created already, `last_chunk_bytes` should be set to the length of the last chunk created. This is so the generator's first output will be enough bytes to fill that chunk up to `chunk_size`.

    Yields:
        Each yield is a chunk of the `content_bytes`. Each chunk is of length (0, `chunk_size`].

    Raises:
        ChunkGeneratorError: If leftover bytes are negative or the previous chunk was invalid.
    """

    if len(content_bytes) <= 0:
        return

    if last_chunk_num_bytes is None:
        bytes_left_in_last_chunk = 0
    else:
        bytes_left_in_last_chunk = chunk_size - last_chunk_num_bytes

    content_num_bytes = len(content_bytes)

    if bytes_left_in_last_chunk < 0:
        raise ChunkGeneratorError(
            "Previous chunk exceeded `chunk_size` (%i > %i)."
            % (bytes_left_in_last_chunk, chunk_size)
        )

    # handle filling the rest of the previous chunk
    total_bytes_yielded = 0
    if bytes_left_in_last_chunk > 0:
        content_bytes_piece = content_bytes[:bytes_left_in_last_chunk]
        yield content_bytes_piece, 0
        total_bytes_yielded += bytes_left_in_last_chunk

        if bytes_left_in_last_chunk > content_num_bytes:
            # if the previous chunk still isn't filled, that means there is no more
            # data to write
            return

    num_chunks_to_create = max(1, int(np.floor(content_num_bytes / chunk_size)))
    start_chunk = 1

    # handle full chunk bytes
    for piece_index, relative_chunk_index in enumerate(
        range(start_chunk, num_chunks_to_create + start_chunk)
    ):
        start = piece_index * chunk_size + bytes_left_in_last_chunk
        end = (piece_index + 1) * chunk_size + bytes_left_in_last_chunk
        content_bytes_piece = content_bytes[start:end]
        if total_bytes_yielded >= content_num_bytes:
            # prevents empty pieces being generated
            break
        yield content_bytes_piece, relative_chunk_index
        total_bytes_yielded += len(content_bytes_piece)

    # handle leftover bytes
    num_leftover_bytes = content_num_bytes - total_bytes_yielded
    if num_leftover_bytes < 0:
        raise ChunkGeneratorError("Leftover bytes should never be negative.")

    if num_leftover_bytes > 0:
        leftover_bytes = content_bytes[end:]
        yield leftover_bytes, relative_chunk_index + 1
