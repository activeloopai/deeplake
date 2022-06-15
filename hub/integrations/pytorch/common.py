from typing import Callable, Dict, List, Optional
from hub.util.iterable_ordered_dict import IterableOrderedDict
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
    return torch.utils.data._utils.collate.default_collate(batch)


def convert_fn(data):
    import torch

    if isinstance(data, IterableOrderedDict):
        return IterableOrderedDict((k, convert_fn(v)) for k, v in data.items())
    if isinstance(data, np.ndarray) and data.size > 0 and isinstance(data[0], str):
        data = data[0]

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
