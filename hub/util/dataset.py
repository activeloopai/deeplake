from typing import Optional, Sequence, List
from hub.util.exceptions import ReadOnlyModeError, TensorDoesNotExistError


def try_flushing(ds):
    try:
        ds.flush()
    except ReadOnlyModeError:
        pass


def map_tensor_keys(dataset, tensor_keys: Optional[Sequence[str]] = None) -> List[str]:
    """Sanitizes tensor_keys if not None, else returns all the keys present in the dataset."""

    if tensor_keys is None:
        tensor_keys = list(dataset.tensors)
    else:
        for t in tensor_keys:
            if t not in dataset.tensors:
                raise TensorDoesNotExistError(t)

        tensor_keys = list(tensor_keys)

    # Get full path in case of groups
    return [dataset.tensors[k].meta.name or dataset.tensors[k].key for k in tensor_keys]
