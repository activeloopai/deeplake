from functools import wraps
from typing import Callable
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
