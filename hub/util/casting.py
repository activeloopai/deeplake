from typing import Union, Sequence
import numpy as np


def get_incompatible_dtype(
    item: Union[np.ndarray, Sequence], dtype: Union[str, np.dtype]
):
    if isinstance(item, (int, float, bool, str)) or hasattr(item, "dtype"):
        return (
            False
            if np.can_cast(item, dtype)
            else getattr(item, "dtype", np.array(item).dtype)
        )
    elif isinstance(item, Sequence):
        return all(map(lambda x: get_incompatible_dtype(x, dtype), item))
    else:
        raise TypeError(
            f"Unexpected object {item}. Expected np.ndarray, int, float, bool, str or Sequence."
        )
