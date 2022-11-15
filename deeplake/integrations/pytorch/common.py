from typing import Callable, Dict, List, Optional
import warnings
from deeplake.util.exceptions import EmptyTensorError
from deeplake.util.iterable_ordered_dict import IterableOrderedDict
from deeplake.core.polygon import Polygons
import numpy as np
import warnings


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
    json_list_tensors = []
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
        if tensor.meta.htype in {"list", "json"}:
            json_list_tensors.append(tensor_name)

    if json_list_tensors:
        warnings.warn(
            f"The following tensors have json or list htype: {json_list_tensors}. Collation of these tensors will fail by default. Ensure that these tensors are either transformed by specifying a transform or a custom collate_fn is specified to handle them."
        )

    return compressed_tensors


def validate_decode_method(decode_method, all_tensor_keys, jpeg_png_compressed_tensors):
    raw_tensors = []
    compressed_tensors = []
    if decode_method is None:
        if len(jpeg_png_compressed_tensors) > 0:
            warnings.warn(
                f"Decode method for tensors {jpeg_png_compressed_tensors} is defaulting to numpy. Please consider specifying a decode_method in .pytorch() that maximizes the data preprocessing speed based on your transformation."
            )
        return raw_tensors, compressed_tensors

    jpeg_png_compressed_tensors_set = set(jpeg_png_compressed_tensors)
    generic_supported_decode_methods = {"numpy", "tobytes"}
    jpeg_png_supported_decode_methods = {"numpy", "tobytes", "pil"}
    for tensor_name, decode_method in decode_method.items():
        if tensor_name not in all_tensor_keys:
            raise ValueError(
                f"decode_method tensor {tensor_name} not found in tensors."
            )
        if tensor_name in jpeg_png_compressed_tensors_set:
            if decode_method not in jpeg_png_supported_decode_methods:
                raise ValueError(
                    f"decode_method {decode_method} not supported for tensor {tensor_name}. Supported methods for this tensor are {jpeg_png_supported_decode_methods}"
                )
        elif decode_method not in generic_supported_decode_methods:
            raise ValueError(
                f"decode_method {decode_method} not supported for tensor {tensor_name}. Supported methods for this tensor are {generic_supported_decode_methods}"
            )
        if decode_method == "tobytes":
            raw_tensors.append(tensor_name)
        elif decode_method == "pil":
            compressed_tensors.append(tensor_name)

    return raw_tensors, compressed_tensors


def get_collate_fn(collate, mode):
    if collate is None and mode == "pytorch":
        return collate_fn
    return collate
