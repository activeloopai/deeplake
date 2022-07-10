from typing import Union
from hub.util.exceptions import InvalidOperationError
from typing import Callable
from functools import wraps
import hub


def invalid_view_op(callable: Callable):
    @wraps(callable)
    def inner(x, *args, **kwargs):
        ds = x if isinstance(x, hub.Dataset) else x.dataset
        # Materialized views have `_is_view = False`. Non materialized views have non trivial indices.
        # Materialized views can be deleted.
        if not getattr(ds, "_allow_view_updates", False) and (
            not x.index.is_trivial()
            or (getattr(ds, "_is_view", False) and callable.__name__ != "delete")
        ):
            raise InvalidOperationError(
                callable.__name__,
                type(x).__name__,
            )
        return callable(x, *args, **kwargs)

    return inner
