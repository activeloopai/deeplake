from typing import Any, List, Tuple
import numpy as np


def calculate_adjacent_runs(L: List[Any]) -> Tuple[List[int], List[Any]]:
    if len(L) == 0:
        return [], []

    if len(L) == 1:
        return [1], L

    counts = []
    elems = []
    running_count = 1
    for i in range(len(L) - 1):
        l = L[i]
        if l == L[i + 1]:
            running_count += 1
        else:
            counts.append(running_count)
            elems.append(l)
            running_count = 1
    counts.append(running_count)
    elems.append(L[-1])
    return counts, elems
