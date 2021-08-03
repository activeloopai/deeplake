from typing import Union, Sequence
import numpy as np


def get_incompatible_dtype(
    samples: Union[np.ndarray, Sequence], dtype: Union[str, np.dtype]
):
    """Check if items in a non-uniform mixed dtype sequence of samples can be safely cast to the given dtype.
    Args:
        samples: Sequence of samples
        dtype: dtype to which samples have to be cast

    Returns:
        None if all samples are compatible. If not, the dtype of the offending item is returned.

    Raises:
        TypeError: if samples is of unexepcted type.
    """
    if isinstance(samples, (int, float, bool, str)) or hasattr(samples, "dtype"):
        return (
            None
            if np.can_cast(getattr(samples, "dtype", samples), dtype)
            else getattr(samples, "dtype", np.array(samples).dtype)
        )
    elif isinstance(samples, Sequence):
        return all(map(lambda x: get_incompatible_dtype(x, dtype), samples))
    else:
        raise TypeError(
            f"Unexpected object {samples}. Expected np.ndarray, int, float, bool, str or Sequence."
        )
