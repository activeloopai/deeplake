import numpy as np


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
