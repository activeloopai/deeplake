from functools import wraps
from typing import Callable, List
import deeplake


def suppress_iteration_warning(callable: Callable):
    @wraps(callable)
    def inner(x, *args, **kwargs):
        iteration_warning_flag = deeplake.constants.SHOW_ITERATION_WARNING
        deeplake.constants.SHOW_ITERATION_WARNING = False
        res = callable(x, *args, **kwargs)
        deeplake.constants.SHOW_ITERATION_WARNING = iteration_warning_flag
        return res

    return inner


def check_if_iteration(indexing_history: List[int], item):
    is_iteration = False
    if len(indexing_history) == 10:
        step = indexing_history[1] - indexing_history[0]
        for i in range(2, len(indexing_history)):
            if indexing_history[i] - indexing_history[i - 1] != step:
                indexing_history.pop(0)
                indexing_history.append(item)
                break
        else:
            is_iteration = True
    else:
        indexing_history.append(item)
    return is_iteration
