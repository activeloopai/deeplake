from typing import Callable, Dict, List, Optional
from deeplake.util.exceptions import EmptyTensorError
from deeplake.util.iterable_ordered_dict import IterableOrderedDict
from deeplake.core.polygon import Polygons
import numpy as np


def collate_fn(batch):
    import torch

    elem = batch[0]
    if isinstance(elem, IterableOrderedDict):
        return IterableOrderedDict(
            (key, collate_fn([d[key] for d in batch])) for key in elem.keys()
        )

    if isinstance(elem, np.ndarray) and elem.size > 0 and isinstance(elem[0], str):
        batch = [it[0] for it in batch]
    elif isinstance(elem, Polygons):
        batch = [it.numpy() for it in batch]
    return torch.utils.data._utils.collate.default_collate(batch)


def convert_fn(data):
    import torch

    if isinstance(data, IterableOrderedDict):
        return IterableOrderedDict((k, convert_fn(v)) for k, v in data.items())
    if isinstance(data, np.ndarray) and data.size > 0 and isinstance(data[0], str):
        data = data[0]
    elif isinstance(data, Polygons):
        data = data.numpy()

    return torch.utils.data._utils.collate.default_convert(data)


class PytorchTransformFunction:
    def __init__(
        self,
        transform_dict: Optional[Dict[str, Optional[Callable]]] = None,
        composite_transform: Optional[Callable] = None,
    ) -> None:
        self.composite_transform = composite_transform
        self.transform_dict = transform_dict

    def __call__(self, data_in: Dict) -> Dict:
        if self.composite_transform is not None:
            return self.composite_transform(data_in)
        elif self.transform_dict is not None:
            data_out = {}
            for tensor, fn in self.transform_dict.items():
                value = data_in[tensor]
                data_out[tensor] = value if fn is None else fn(value)
            data_out = IterableOrderedDict(data_out)
            return data_out
        return data_in


def check_tensors(dataset, tensors):
    compressed_tensors = []
    supported_compressions = {"png", "jpeg"}
    for tensor_name in tensors:
        tensor = dataset._get_tensor_from_root(tensor_name)
        if len(tensor) == 0:
            raise EmptyTensorError(
                f" the dataset has an empty tensor {tensor_name}, pytorch dataloader can't be created."
                f" Please either populate the tensor or pass tensors argument to .pytorch that excludes this"
                f" tensor."
            )
        if tensor.meta.sample_compression in supported_compressions:
            compressed_tensors.append(tensor_name)
    return compressed_tensors


def remove_intersections(compressed_tensors: List[str], raw_tensors: List[str]):
    compressed_tensors = [
        tensor for tensor in compressed_tensors if tensor not in raw_tensors
    ]
    raw_tensors.extend(compressed_tensors)
    return compressed_tensors, raw_tensors
