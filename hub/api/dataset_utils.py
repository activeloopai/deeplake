import numpy as np
import sys
from hub.exceptions import ModuleNotInstalledException


def slice_split(slice_):
    """Splits a slice into subpath and list of slices"""
    path = ""
    list_slice = []
    for sl in slice_:
        if isinstance(sl, str):
            path += sl if sl.startswith("/") else "/" + sl
        elif isinstance(sl, (int, slice)):
            list_slice.append(sl)
        else:
            raise TypeError(
                "type {} isn't supported in dataset slicing".format(type(sl))
            )
    return path, list_slice


def create_numpy_dict(dataset, index):
    numpy_dict = {}
    for path in dataset._tensors.keys():
        d = numpy_dict
        split = path.split("/")
        for subpath in split[1:-1]:
            if subpath not in d:
                d[subpath] = {}
            d = d[subpath]
        d[split[-1]] = dataset[path, index].numpy()
    return numpy_dict


def get_value(value):
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    elif isinstance(value, list):
        for i in range(len(value)):
            if isinstance(value[i], np.ndarray) and value[i].shape == ():
                value[i] = value[i].item()
    return value


def str_to_int(assign_value, tokenizer):
    if isinstance(assign_value, bytes):
        try:
            assign_value = assign_value.decode("utf-8")
        except Exception:
            raise ValueError(
                "Bytes couldn't be decoded to string. Other encodings of bytes are currently not supported"
            )
    if (
        isinstance(assign_value, np.ndarray) and assign_value.dtype.type is np.bytes_
    ) or (isinstance(assign_value, list) and isinstance(assign_value[0], bytes)):
        assign_value = [item.decode("utf-8") for item in assign_value]
    if tokenizer is not None:
        if "transformers" not in sys.modules:
            raise ModuleNotInstalledException("transformers")
        import transformers

        global transformers
        tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
        assign_value = (
            np.array(tokenizer(assign_value, add_special_tokens=False)["input_ids"])
            if isinstance(assign_value, str)
            else assign_value
        )
        if (
            isinstance(assign_value, list)
            and assign_value
            and isinstance(assign_value[0], str)
        ):
            assign_value = [
                np.array(tokenizer(item, add_special_tokens=False)["input_ids"])
                for item in assign_value
            ]
    else:
        assign_value = (
            np.array([ord(ch) for ch in assign_value])
            if isinstance(assign_value, str)
            else assign_value
        )
        if (
            isinstance(assign_value, list)
            and assign_value
            and isinstance(assign_value[0], str)
        ):
            assign_value = [np.array([ord(ch) for ch in item]) for item in assign_value]
    return assign_value
