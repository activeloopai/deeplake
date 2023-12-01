from deeplake.core.vectorstore.dataset_handlers.client_side_dataset_handler import (
    ClientSideDH,
)


def get_dataset_handler(*args, **kwargs):
    # TODO: Use this logic when managed db will be ready
    # runtime = kwargs.get("runtime", None)
    # if runtime and runtime.get("tensor_db", True):
    #     return ClientSideDH(*args, **kwargs)
    # else:
    #     return ClientSideDH(*args, **kwargs)
    return ClientSideDH(*args, **kwargs)
