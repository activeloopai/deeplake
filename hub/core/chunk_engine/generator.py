import numpy as np


def _assert_valid_piece(piece, chunk_size):
    assert len(piece) > 0
    assert len(piece) <= chunk_size


def chunk(content_bytes, previous_num_bytes, chunk_size):
    """
    Generator function that chunks bytes.

    Args:
        content_bytes(bytes): Bytes object with the data to be chunked.
        previous_num_bytes(int): The number of bytes in the previous chunk. This is helpful for filling the previous chunk before starting a new one.
        chunk_size(int): Each individual chunk will be assigned this many bytes maximum.

    Yields:
        Each yield is a chunk of the `content_bytes`. Each chunk is of length (0, `chunk_size`].
    """

    if len(content_bytes) <= 0:
        return

    if previous_num_bytes is None:
        bytes_left = 0
    else:
        bytes_left = chunk_size - previous_num_bytes

    content_num_bytes = len(content_bytes)

    if bytes_left < 0:
        # TODO place in exceptions.py & update docstring
        raise Exception(
            "previous chunk exceeded chunk_size. %i > %i" % (bytes_left, chunk_size)
        )

    # handle filling the rest of the previous chunk
    total_bytes_yielded = 0
    if bytes_left > 0:
        content_bytes_piece = content_bytes[:bytes_left]
        _assert_valid_piece(content_bytes_piece, chunk_size)
        yield content_bytes_piece, 0
        total_bytes_yielded += bytes_left

        if bytes_left > content_num_bytes:
            # if the previous chunk still isn't filled, that means there is no more
            # data to write
            return

    num_chunks_to_create = max(1, int(np.floor(content_num_bytes / chunk_size)))
    start_chunk = 1

    # handle full chunk bytes
    for piece_index, relative_chunk_index in enumerate(
        range(start_chunk, num_chunks_to_create + start_chunk)
    ):
        start = piece_index * chunk_size + bytes_left
        end = (piece_index + 1) * chunk_size + bytes_left
        content_bytes_piece = content_bytes[start:end]
        if total_bytes_yielded >= content_num_bytes:
            # prevents empty pieces being generated
            break
        _assert_valid_piece(content_bytes_piece, chunk_size)
        yield content_bytes_piece, relative_chunk_index
        total_bytes_yielded += len(content_bytes_piece)

    # handle leftover bytes
    num_leftover_bytes = content_num_bytes - total_bytes_yielded
    if num_leftover_bytes < 0:
        # TODO place in exceptions.py & update docstring
        raise Exception("leftover bytes should never be negative")

    if num_leftover_bytes > 0:
        leftover_bytes = content_bytes[end:]
        _assert_valid_piece(leftover_bytes, chunk_size)
        yield leftover_bytes, relative_chunk_index + 1
