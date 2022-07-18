from typing import Union
from hub.util.exceptions import InvalidOperationError
from typing import Callable
from functools import wraps
import hub


def invalid_view_op(callable: Callable):
    @wraps(callable)
    def inner(x, *args, **kwargs):
        ds = x if isinstance(x, hub.Dataset) else x.dataset
        if not getattr(ds, "_allow_view_updates", False):
            is_del = callable.__name__ == "delete"
            managed_view = hasattr(ds, "_view_entry")
            has_vds = getattr(x, "_vds", False)
            is_view = not x.index.is_trivial() or has_vds or managed_view
            if is_view and not (is_del and (has_vds or managed_view)):
                raise InvalidOperationError(
                    callable.__name__,
                    type(x).__name__,
                )
        return callable(x, *args, **kwargs)

    return inner
