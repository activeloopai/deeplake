from typing import Tuple
import numpy as np


def _filter_ones(shape: Tuple[int]):
    out = tuple([x for x in shape if x != 1])
    if len(out) <= 0:
        return (1,)
    return out


def normalize_and_batchify_shape(shape: Tuple[int], batched: bool) -> Tuple[int]:
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
        array (np.ndarray): Array that will have it's shape normalized/batchified.
        batched (bool): If True, `array.shape[0]` is assumed to be the batch axis. If False,
            an axis will be added such that `array.shape[0] == 1`.

    Returns:
        np.ndarray: Array with a guarenteed batch dimension. `out_array.shape[1:]` will always be > 1.
            `out_array.shape[0]` may be >= 1.
    """

    if len(shape) < 1:
        raise ValueError("Empty shape cannot be normalized.")

    if batched:
        if len(shape) < 2:
            raise ValueError("A shape with length < 2 cannot be batched. Shape: {}".format(shape))
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
