from hub.util.exceptions import InvalidOperationError
from typing import Callable
from functools import wraps
import hub


def no_view(callable: Callable):
    @wraps(callable)
    def inner(x, *args, **kwargs):
        func = callable.__name__
        if not x.index.is_trivial():
            if func == "read_only":
                if not x._read_only_set:
                    return callable(x, *args, **kwargs)
            raise InvalidOperationError(
                func,
                "Dataset" if isinstance(x, hub.Dataset) else "Tensor",
            )
        return callable(x, *args, **kwargs)

    return inner
