from typing import List, Union
from functools import partial
import numpy as np


def slice_array_list(arrays: List[np.ndarray], index: Union[slice, int, List[int]]):
    if isinstance(index, int):
        x, y = index_array_list(arrays, index)
        return arrays[x][y]
    if isinstance(index, list):
        return list(map(partial(slice_array_list, arrays=arrays), index))
    n = sum(map(len, arrays))
    start = index.start or 0
    stop = index.stop or n
    step = index.step or 1
    if step != 1:
        raise NotImplementedError("Stepped indexing is not supported here yet.")
    if start < 0:
        start += n
    if stop < 0:
        stop += n
    if start >= n:
        return []
    if stop <= start:
        return []
    start_xy = index_array_list(arrays, start)
    stop_xy = index_array_list(arrays, stop - 1)
    if start_xy[0] == stop_xy[0]:
        return arrays[start_xy[0]][start_xy[1] : stop_xy[1] + 1]
    else:
        return [
            arrays[start_xy[0]][start_xy[1:]],
            *arrays[start_xy[0] + 1 : stop_xy[0]],
            arrays[stop_xy[0]][: stop_xy[1] + 1],
        ]


def index_array_list(arrays: List[np.ndarray], idx: int):
    csum = np.cumsum(list(map(len, arrays)))
    x = np.searchsorted(csum, idx + 1)
    y = idx - csum[x - 1] if x else idx
    return (x, y)


def reverse_array_list_inplace(arrays: List[np.ndarray]):
    arrays.reverse()
    for i in range(len(arrays)):
        arrays[i] = arrays[i][::-1]


def reverse_array_list(arrays: List[np.ndarray]):
    return [a[::-1] for a in arrays[::-1]]
