import numpy as np

from typing import Tuple


def normalize_and_batchify_shape(array: np.ndarray, batched: bool) -> np.ndarray:
    """Remove all `array.shape` axes with a value of 1 & add a batch dimension if needed.

    Example 1:
        input_array.shape = (10, 1, 5)
        batched = False
        output_array.shape = (1, 10, 5)  # batch axis is added

    Example 2:
        input_array.shape = (1, 100, 1, 1, 3)
        batched = True
        output_array.shape = (1, 100, 3)  # batch axis is preserved

    Args:
        array (np.ndarray): Array that will have it's shape normalized/batchified.
        batched (bool): If True, `array.shape[0]` is assumed to be the batch axis. If False,
            an axis will be added such that `array.shape[0] == 1`.

    Returns:
        np.ndarray: Array with a guarenteed batch dimension. `out_array.shape[1:]` will always be > 1.
            `out_array.shape[0]` may be >= 1.
    """

    if batched:
        # Don't squeeze the primary axis, even if it's 1
        squeeze_axes = tuple(i + 1 for i, s in enumerate(array.shape[1:]) if s == 1)
        array = array.squeeze(squeeze_axes)
    else:
        array = array.squeeze()
        array = np.expand_dims(array, axis=0)

    # If we squeezed everything except the primary axis, append one dimension of length 1
    if len(array.shape) == 1:
        array = np.expand_dims(array, axis=1)
    return array


def get_random_array(shape: Tuple[int], dtype: str) -> np.ndarray:
    dtype = dtype.lower()

    if "int" in dtype:
        low = np.iinfo(dtype).min
        high = np.iinfo(dtype).max
        return np.random.randint(low=low, high=high, size=shape, dtype=dtype)

    if "float" in dtype:
        # get float16 because np.random.uniform doesn't support the `dtype` argument.
        low = np.finfo("float16").min
        high = np.finfo("float16").max
        return np.random.uniform(low=low, high=high, size=shape).astype(dtype)

    if "bool" in dtype:
        a = np.random.uniform(size=shape)
        return a > 0.5

    raise ValueError("Dtype %s not supported." % dtype)