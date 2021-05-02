from hub.core.chunk_engine.generator import chunk
from hub.core.chunk_engine.tests.util import (
    make_dummy_byte_array,
    get_random_chunk_size,
    get_random_num_samples,
    get_random_partial,
)


TRIALS_PER_TEST = 10


def test_perfect_fit():
    """This test is for the edge case where: (num_bytes % chunk_size == 0). In other words, bytes fit perfectly into chunks with none left over."""

    for _ in range(TRIALS_PER_TEST):
        chunk_size = get_random_chunk_size()
        num_samples = get_random_num_samples()

        content = make_dummy_byte_array(chunk_size * num_samples)
        gen = chunk(content, previous_num_bytes=None, chunk_size=chunk_size)

        piece_count = 0
        for piece, relative_chunk_index in gen:
            expected_start = piece_count * chunk_size
            expected_end = (piece_count + 1) * chunk_size
            expected_bytes = content[expected_start:expected_end]

            assert (
                piece_count + 1 == relative_chunk_index
            ), "relative chunk index needs to start @ 1 since 1 means to create a new chunk"
            assert expected_bytes == piece, "expected bytes don't match returned piece"

            piece_count += 1

        assert (
            piece_count == num_samples
        ), "perfect fit means the number of pieces generated should be equal to the number of samples"


def test_first_partial_chunk():
    """This test is for the edge case where: (num_bytes < chunk_size). In other words, bytes only partially fill 1 chunk."""
    for _ in range(TRIALS_PER_TEST):
        chunk_size = get_random_chunk_size()

        # divide by 2 so part 2 doesn't fill all the way
        content_length = get_random_partial(chunk_size // 2)

        assert content_length < (chunk_size // 2), "test invalid"

        content = make_dummy_byte_array(content_length)
        gen = chunk(content, previous_num_bytes=None, chunk_size=chunk_size)

        piece_count = 0
        for piece, relative_chunk_index in gen:
            assert (
                piece_count + 1 == relative_chunk_index
            ), "relative chunk index needs to start @ 1 since 1 means to create a new chunk"
            assert piece == content
            piece_count += 1

        assert piece_count == 1

        # part 2: try to write to a chunk (the one created in the previous part)
        # that has been partially filled. but this doesn't fill the chunk all the way
        # it is still partial
        # basically, the edge case is: (content_length + old_content_length) < chunk_size)
        old_content_length = content_length
        content_length = get_random_partial(chunk_size // 2)
        assert old_content_length + content_length < chunk_size

        content = make_dummy_byte_array(content_length)
        gen = chunk(
            content, previous_num_bytes=old_content_length, chunk_size=chunk_size
        )

        piece_count = 0
        for piece, relative_chunk_index in gen:
            assert (
                piece_count == relative_chunk_index
            ), "relative chunk index needs to start @ 0 since 0 means to append to the previous chunk"
            assert piece == content
            piece_count += 1

        assert piece_count == 1


def test_nth_partial_chunk():
    """This test is for the edge case where: ((num_bytes > chunk_size) and (num_bytes % chunk_size != 0)). In other words, bytes fill at least 1 chunk fully, but the last chunk is only partially filled."""
    for _ in range(TRIALS_PER_TEST):
        chunk_size = get_random_chunk_size()
        n = get_random_num_samples()
        partial_length = get_random_partial(chunk_size)
        content_length = partial_length + (chunk_size * n)
        assert partial_length < chunk_size, "test invalid"
        assert content_length % chunk_size != 0

        content = make_dummy_byte_array(content_length)
        gen = chunk(content, previous_num_bytes=None, chunk_size=chunk_size)

        piece_count = 0
        previous_num_bytes = None
        for piece, relative_chunk_index in gen:
            expected_start = piece_count * chunk_size
            expected_end = (piece_count + 1) * chunk_size
            expected_bytes = content[expected_start:expected_end]

            assert (
                piece_count + 1 == relative_chunk_index
            ), "relative chunk index needs to start @ 1 since 1 means to create a new chunk"
            assert piece == expected_bytes, (
                "piece # %i doesn't match expected content" % piece_count
            )

            previous_num_bytes = len(piece)
            piece_count += 1

        assert piece_count == n + 1  # plus 1 because partial
        assert previous_num_bytes == partial_length, "invalid test"

        # part 2: need to add more content after the nth partial
        old_partial_length = partial_length
        old_n = n
        n = get_random_num_samples()
        partial_length = get_random_partial(chunk_size)
        content_length = partial_length + (chunk_size * n)
        assert partial_length < chunk_size, "test invalid"
        assert content_length % chunk_size != 0

        content = make_dummy_byte_array(content_length)
        gen = chunk(
            content, previous_num_bytes=previous_num_bytes, chunk_size=chunk_size
        )

        piece_count = 0
        bytes_left = chunk_size - previous_num_bytes
        is_first = True
        for piece, relative_chunk_index in gen:
            if is_first:
                expected_bytes = content[:bytes_left]
            else:
                expected_start = (piece_count - 1) * chunk_size + bytes_left
                expected_end = (piece_count) * chunk_size + bytes_left
                expected_bytes = content[expected_start:expected_end]

            assert (
                piece_count == relative_chunk_index
            ), "relative chunk index needs to start @ 0 since 0 means to append to the previous chunk"
            assert piece == expected_bytes, (
                "piece # %i doesn't match expected content" % piece_count
            )

            piece_count += 1
            is_first = False

        extra_chunks = 1 if partial_length + old_partial_length <= chunk_size else 2
        assert piece_count == n + extra_chunks
