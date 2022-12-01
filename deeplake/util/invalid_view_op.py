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
            is_ds_view = (
                ds._view_base
                and not (
                    ds.index.is_trivial()
                    and ds.version_state["commit_node"].is_head_node
                )
                and not ds.group_index
            )
            is_view = not x.index.is_trivial() or has_vds or managed_view or is_ds_view
            if is_view and not (is_del and (has_vds or managed_view)):
                raise InvalidOperationError(
                    callable.__name__,
                    type(x).__name__,
                )
        return callable(x, *args, **kwargs)

    return inner
