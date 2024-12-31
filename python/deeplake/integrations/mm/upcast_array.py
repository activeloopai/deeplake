import numpy as np
from typing import Union


def upcast_array(arr: Union[np.ndarray, bytes]):
    if isinstance(arr, list):
        return [upcast_array(a) for a in arr]
    if isinstance(arr, np.ndarray):
        if arr.dtype == np.uint16:
            return arr.astype(np.int32)
        if arr.dtype == np.uint32:
            return arr.astype(np.int64)
        if arr.dtype == np.uint64:
            return arr.astype(np.int64)
    return arr
