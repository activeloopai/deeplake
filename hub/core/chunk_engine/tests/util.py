from copy import deepcopy

import numpy as np


def make_dummy_byte_array(length: int) -> bytearray:
    """Generate a random bytearray of the provided length."""
    content = bytearray()
    a = np.random.randint(128, size=length)
    content.extend(a.tolist())
    assert len(content) == length
    return content


def get_random_chunk_size() -> int:
    return np.random.choice([8, 256, 1024, 4096])


def get_random_num_samples() -> int:
    return np.random.randint(1, 300)


def get_random_partial(chunk_size: int) -> int:
    return np.random.randint(1, chunk_size - 1)


def assert_valid_chunk(chunk: bytes, chunk_size: int):
    assert len(chunk) > 0
    assert len(chunk) <= chunk_size
