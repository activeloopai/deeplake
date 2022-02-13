from typing import Union
from hub.core.dataset import Dataset
from hub.core.tensor import Tensor
from hub.util.exceptions import InvalidOperationError
from typing import Callable
from functools import wraps
import hub


def invalid_view_op(callable: Callable):
    @wraps(callable)
    def inner(x: Union[Dataset, Tensor], *args, **kwargs):
        if not x.index.is_trivial():
            raise InvalidOperationError(
                callable.__name__,
                type(x).__name__,
            )
        return callable(x, *args, **kwargs)

    return inner
