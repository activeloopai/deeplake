from typing import Union
from deeplake.util.exceptions import InvalidOperationError
from typing import Callable
from functools import wraps
import deeplake


def invalid_view_op(callable: Callable):
    @wraps(callable)
    def inner(x, *args, **kwargs):
        ds = x if isinstance(x, deeplake.Dataset) else x.dataset
        if not ds.__dict__.get("_allow_view_updates"):
            is_del = callable.__name__ == "delete"
            managed_view = "_view_entry" in ds.__dict__
            has_vds = "_vds" in ds.__dict__
            is_view = not x.index.is_trivial() or has_vds or managed_view
            if is_view and not (is_del and (has_vds or managed_view)):
                raise InvalidOperationError(
                    callable.__name__,
                    type(x).__name__,
                )
        return callable(x, *args, **kwargs)

    return inner
