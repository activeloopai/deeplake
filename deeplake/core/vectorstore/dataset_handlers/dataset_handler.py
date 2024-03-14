from deeplake.core.vectorstore.dataset_handlers.embedded_dataset_handler import (
    EmbeddedDH,
)
from deeplake.core.vectorstore.dataset_handlers.managed_dataset_handler import (
    ManagedDH,
)


def get_dataset_handler(*args, **kwargs):
    runtime = kwargs.get("runtime", None)
    if runtime and runtime.get("tensor_db", True):
        # TODO: change to ManagedSideDH when it's ready
        return ManagedDH(*args, **kwargs)
    else:
        return EmbeddedDH(*args, **kwargs)
