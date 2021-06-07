import pytest

import numpy as np

from hub.util.array import normalize_and_batchify_array_shape

from typing import Tuple


@pytest.mark.parametrize(
    "shape,expected_shape,batched",
    (
        # batched
        [(1, 100), (1, 100), True],
        [(100, 1), (100, 1), True],
        [(100, 100), (100, 100), True],
        [(10, 224, 224, 3), (10, 224, 224, 3), True],
        [(10, 3, 224, 224), (10, 3, 224, 224), True],
        [(10, 3, 224, 224, 1), (10, 3, 224, 224), True],
        [(10, 3, 224, 224, 1, 1, 1, 1), (10, 3, 224, 224), True],
        [(1, 1, 1, 1, 1, 1, 1), (1, 1), True],
        [(1, 1), (1, 1), True],
        # not batched
        [(1,), (1, 1), False],
        [(100,), (1, 100), False],
        [(1, 100), (1, 100), False],
        [(100, 1), (1, 100), False],
        [(100, 100), (1, 100, 100), False],
        [(3, 224, 224, 1, 1, 1, 1), (1, 3, 224, 224), False],
        [(10, 3, 224, 224, 1, 1, 1, 1), (1, 10, 3, 224, 224), False],
        [(1, 1), (1, 1), False],
        [(1, 1, 1, 1, 1, 1, 1), (1, 1), False],
    ),
)
def test_normalize_and_batchify_shape(
    shape: Tuple[int], expected_shape: Tuple[int], batched: bool
):
    a = np.random.uniform(size=shape)
    normal_a = normalize_and_batchify_array_shape(a, batched)
    assert normal_a.shape == expected_shape
    np.testing.assert_array_equal(a.flatten(), normal_a.flatten())
