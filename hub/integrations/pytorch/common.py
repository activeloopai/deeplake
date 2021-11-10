from typing import Callable, Dict, List, Optional, Sequence
from hub.util.exceptions import TensorDoesNotExistError
from hub.util.iterable_ordered_dict import IterableOrderedDict


def collate_fn(batch):
    import torch

    elem = batch[0]
    if isinstance(elem, IterableOrderedDict):
        return IterableOrderedDict(
            (key, collate_fn([d[key] for d in batch])) for key in elem.keys()
        )
    return torch.utils.data._utils.collate.default_collate(batch)


def convert_fn(data):
    import torch

    elem_type = type(data)
    if isinstance(data, IterableOrderedDict):
        return IterableOrderedDict((k, convert_fn(v)) for k, v in data.items())
    return torch.utils.data._utils.collate.default_convert(data)


class PytorchTransformFunction:
    def __init__(
        self, transform: Dict[str, Optional[Callable]], tensors: List[str]
    ) -> None:
        self.transform = transform
        for tensor in transform:
            if tensor not in tensors:
                raise ValueError(f"Invalid transform. Tensor {tensor} not found.")

    def __call__(self, data_in: Dict) -> Dict:
        data_out = {}
        for tensor, fn in self.transform.items():
            value = data_in[tensor]
            data_out[tensor] = value if fn is None else fn(value)
        return data_out


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
    return [dataset.tensors[k].key for k in tensor_keys]
