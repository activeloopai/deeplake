from typing import Tuple
import numpy as np


def _filter_ones(shape: Tuple):
    """Removes all 1s from `shape`. If ALL values in `shape` are 1s, `(1,)` is returned as the shape."""

    out = tuple(x for x in shape if x != 1)
    if not out:
        return (1,)
    return out


def normalize_and_batchify_shape(
    shape: Tuple[int, ...], batched: bool
) -> Tuple[int, ...]:
    """Remove all 1s from `shape`. If `batched`, shape[0] is preserved, otherwise a batch axis is prepended.

    Example 1:
        shape = (10, 1, 5)
        batched = False
        shape = (1, 10, 5)  # batch axis is added

    Example 2:
        shape = (1, 100, 1, 1, 3)
        batched = True
        shape = (1, 100, 3)  # batch axis is preserved

    Args:
        shape (tuple): Shape that will be normalized/batchified.
        batched (bool): If True, `shape[0]` is assumed to be the batch axis. If False,
            an axis will be added such that `shape[0] == 1`.

    Raises:
        ValueError: If an invalid `shape` is provided.

    Returns:
        tuple: All entries in `shape[1:]` will be > 1. Shape will have a guarenteed batch axis (`shape[0] >= 1`).
    """

    if not shape:
        raise ValueError("Empty shape cannot be normalized.")

    if batched:
        if len(shape) == 2:
            return shape

        norm_sample_shape = _filter_ones(shape[1:])
        norm_sample_shape = (shape[0],) + norm_sample_shape
    else:

        if len(shape) == 1:
            norm_sample_shape = shape
        else:
            norm_sample_shape = _filter_ones(shape)

        norm_sample_shape = (1,) + norm_sample_shape

    return norm_sample_shape


def normalize_and_batchify_array_shape(array: np.ndarray, batched: bool) -> np.ndarray:
    """Reshape `array` with the output shape of `normalize_and_batchify_shape`."""

    return array.reshape(normalize_and_batchify_shape(array.shape, batched))
