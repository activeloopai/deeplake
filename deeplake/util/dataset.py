from typing import Optional, Sequence, List
from deeplake.util.exceptions import ReadOnlyModeError, TensorDoesNotExistError


def try_flushing(ds):
    try:
        ds.flush()
    except ReadOnlyModeError:
        pass


def map_tensor_keys(dataset, tensor_keys: Optional[Sequence[str]] = None) -> List[str]:
    """Sanitizes tensor_keys if not None, else returns all the keys present in the dataset."""

    tensors = dataset.tensors

    if tensor_keys is None:
        tensor_keys = list(tensors)
    else:
        for t in tensor_keys:
            if t not in tensors:
                raise TensorDoesNotExistError(t)

        tensor_keys = list(tensor_keys)

    # Get full path in case of groups
    return [tensors[k].meta.name or tensors[k].key for k in tensor_keys]
