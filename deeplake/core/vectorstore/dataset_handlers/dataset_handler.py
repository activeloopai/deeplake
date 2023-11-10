from deeplake.core.vectorstore.dataset_handlers.client_side_dataset_handler import (
    ClientSideDH,
)
from deeplake.core.vectorstore.dataset_handlers.managed_side_dataset_handler import (
    ManagedSideDH,
)


def get_dataset_handler(*args, **kwargs):
    runtime = kwargs.get("runtime", None)
    if runtime and runtime.get("tensor_db", False):
        return ManagedSideDH(*args, **kwargs)
    else:
        return ClientSideDH(*args, **kwargs)
