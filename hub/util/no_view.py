from hub.util.exceptions import NoViewError
from typing import Callable, Union
import hub


def no_view(callable: Callable):
    def inner(x, *args, **kwargs):
        if not x.index.is_trivial():
            raise NoViewError(
                callable.__name__,
                "Dataset" if isinstance(x, hub.Dataset) else "Tensor",
            )
        return callable(x, *args, **kwargs)

    return inner
