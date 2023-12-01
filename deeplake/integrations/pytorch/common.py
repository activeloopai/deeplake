from typing import Callable, Dict, Optional
import warnings
from deeplake.util.class_label import convert_to_text
from deeplake.util.exceptions import EmptyTensorError
from deeplake.util.iterable_ordered_dict import IterableOrderedDict
from deeplake.util.keys import get_sample_info_tensor_key
from deeplake.util.object_3d.mesh import parse_mesh_to_dict
from deeplake.util.object_3d.point_cloud import parse_point_cloud_to_dict
from deeplake.core.polygon import Polygons
import numpy as np
import warnings


def collate_fn(batch):
    import torch
    from torch.utils.data._utils.collate import default_collate

    elem = batch[0]
    if isinstance(elem, IterableOrderedDict):
        return IterableOrderedDict(
            (key, collate_fn([d[key] for d in batch])) for key in elem.keys()
        )

    if isinstance(elem, np.ndarray) and elem.size > 0 and isinstance(elem[0], str):
        if elem.dtype == object:
            return [it.tolist() for it in batch]
        else:
            batch = [it[0] for it in batch]
    elif isinstance(elem, (tuple, list)) and len(elem) > 0 and isinstance(elem[0], str):
        batch = [it[0] for it in batch]
    elif isinstance(elem, Polygons):
        batch = [it.numpy() for it in batch]
    elif isinstance(elem, list) and all(
        map(lambda e: isinstance(e, np.ndarray), elem)
    ):  # special case for video query api
        if (
            len(elem) > 0
            and len(elem[0].shape) > 1
            and elem[0].shape[1]
            not in [
                2,
                3,
            ]
        ):  # checking whether it is not a polygon
            elem_type = type(elem)
            return [
                elem_type([torch.tensor(item) for item in sample]) for sample in batch
            ]
    return default_collate(batch)


def convert_fn(data):
    from torch.utils.data._utils.collate import default_convert

    if isinstance(data, IterableOrderedDict):
        return IterableOrderedDict((k, convert_fn(v)) for k, v in data.items())
    if isinstance(data, np.ndarray) and data.size > 0 and isinstance(data[0], str):
        if data.dtype == object:
            return data.tolist()
        else:
            data = data[0]
    elif isinstance(data, Polygons):
        data = data.numpy()

    return default_convert(data)


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


def check_tensors(dataset, tensors, verbose=True):
    jpeg_png_compressed_tensors = []
    json_tensors = []
    list_tensors = []
    tag_tensors = []
    supported_image_compressions = {"png", "jpeg"}
    for tensor_name in tensors:
        tensor = dataset._get_tensor_from_root(tensor_name)
        if len(tensor) == 0:
            raise EmptyTensorError(
                f" the dataset has an empty tensor {tensor_name}, pytorch dataloader can't be created."
                f" Please either populate the tensor or pass tensors argument to .pytorch that excludes this"
                f" tensor."
            )
        meta = tensor.meta
        if meta.sample_compression in supported_image_compressions:
            jpeg_png_compressed_tensors.append(tensor_name)
        elif meta.htype == "json":
            json_tensors.append(tensor_name)
        elif meta.htype == "list":
            list_tensors.append(tensor_name)
        elif meta.htype == "tag":
            tag_tensors.append(tensor_name)

    if verbose and (json_tensors or list_tensors):
        json_list_tensors = set(json_tensors + list_tensors)
        warnings.warn(
            f"The following tensors have json or list htype: {json_list_tensors}. Collation of these tensors will fail by default. Ensure that these tensors are either transformed by specifying a transform or a custom collate_fn is specified to handle them."
        )

    list_tensors += tag_tensors

    return jpeg_png_compressed_tensors, json_tensors, list_tensors


def validate_decode_method(
    decode_method,
    all_tensor_keys,
    jpeg_png_compressed_tensors,
    json_tensors,
    list_tensors,
):
    raw_tensors = []
    pil_compressed_tensors = []
    data_tensors = []
    if decode_method is None:
        if len(jpeg_png_compressed_tensors) > 0:
            warnings.warn(
                f"Decode method for tensors {jpeg_png_compressed_tensors} is defaulting to numpy. Please consider specifying a decode_method in .pytorch() that maximizes the data preprocessing speed based on your transformation."
            )
        return (
            raw_tensors,
            pil_compressed_tensors,
            json_tensors,
            list_tensors,
            data_tensors,
        )

    jpeg_png_compressed_tensors_set = set(jpeg_png_compressed_tensors)
    json_list_tensors_set = set(json_tensors + list_tensors)
    generic_supported_decode_methods = {"numpy", "tobytes", "data"}
    jpeg_png_supported_decode_methods = {"numpy", "tobytes", "pil", "data"}
    json_list_supported_decode_methods = {"numpy", "data"}
    for tensor_name, decode_method in decode_method.items():
        if tensor_name not in all_tensor_keys:
            raise ValueError(
                f"tensor {tensor_name} specified in decode_method not found in tensors."
            )
        if tensor_name in jpeg_png_compressed_tensors_set:
            if decode_method not in jpeg_png_supported_decode_methods:
                raise ValueError(
                    f"decode_method {decode_method} not supported for tensor {tensor_name}. Supported methods for this tensor are {jpeg_png_supported_decode_methods}"
                )
        elif tensor_name in json_list_tensors_set:
            if decode_method not in json_list_supported_decode_methods:
                raise ValueError(
                    f"decode_method {decode_method} not supported for tensor {tensor_name}. Supported methods for this tensor are {json_list_supported_decode_methods}"
                )
        elif decode_method not in generic_supported_decode_methods:
            raise ValueError(
                f"decode_method {decode_method} not supported for tensor {tensor_name}. Supported methods for this tensor are {generic_supported_decode_methods}"
            )
        if decode_method == "tobytes":
            raw_tensors.append(tensor_name)
        elif decode_method == "pil":
            pil_compressed_tensors.append(tensor_name)
        elif decode_method == "data":
            data_tensors.append(tensor_name)
    return raw_tensors, pil_compressed_tensors, json_tensors, list_tensors, data_tensors


def find_additional_tensors_and_info(dataset, data_tensors):
    sample_info_htypes = {
        "image",
        "image.rgb",
        "image.gray",
        "dicom",
        "nifti",
        "point_cloud",
        "mesh",
        "video",
    }
    tensor_info_htypes = {"class_label"}

    sample_info_tensors = set()
    tensor_info_tensors = set()
    for tensor_name in data_tensors:
        tensor = dataset._get_tensor_from_root(tensor_name)
        htype = tensor.htype
        if htype in sample_info_htypes:
            info_tensor_name = get_sample_info_tensor_key(tensor_name)
            if tensor._sample_info_tensor:
                sample_info_tensors.add(info_tensor_name)
        if htype in tensor_info_htypes:
            tensor_info_tensors.add(tensor_name)
        if htype == "video":
            raise NotImplementedError(
                "data decode method for video tensors isn't supported yet."
            )
    return sample_info_tensors, tensor_info_tensors


def get_htype_ndim_tensor_info_dicts(dataset, data_tensors, tensor_info_tensors):
    htype_dict = {}
    ndim_dict = {}
    tensor_info_dict = {}
    for tensor_name in data_tensors:
        tensor = dataset._get_tensor_from_root(tensor_name)
        htype_dict[tensor_name] = tensor.htype
        ndim_dict[tensor_name] = tensor.ndim
        if tensor_name in tensor_info_tensors:
            tensor_info_dict[tensor_name] = tensor.info._info
    return htype_dict, ndim_dict, tensor_info_dict


def convert_sample_to_data(sample: dict, htype_dict, ndim_dict, tensor_info_dict):
    for tensor_name in htype_dict.keys():
        value = sample[tensor_name]
        htype = htype_dict[tensor_name]
        ndim = ndim_dict[tensor_name]
        tensor_info = tensor_info_dict.get(tensor_name)
        sample_info = sample.pop(get_sample_info_tensor_key(tensor_name), None)
        sample[tensor_name] = convert_value_to_data(
            value, tensor_info, sample_info, htype, ndim
        )


def convert_value_to_data(value, tensor_info, sample_info, htype, ndim):
    if htype in {"text", "json"}:
        if not isinstance(value, str):
            value = value[0]
        return {"value": value}
    elif htype == "video":
        raise NotImplementedError
    if htype == "class_label":
        labels = value
        data = {"value": labels}
        class_names = tensor_info.get("class_names") if tensor_info else None
        if class_names:
            data["text"] = convert_to_text(labels, class_names)
        return data
    if htype in ("image", "image.rgb", "image.gray", "dicom", "nifti"):
        return {
            "value": value,
            "sample_info": sample_info[0] or {},
        }
    elif htype == "point_cloud":
        return parse_point_cloud_to_dict(value, ndim, sample_info)
    elif htype == "mesh":
        return parse_mesh_to_dict(value, sample_info)
    else:
        return {"value": value}
