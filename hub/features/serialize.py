import copy

from hub.features.features import Primitive, Tensor, FeatureDict
# from hub.features.class_label import ClassLabel
# from hub.features.sequence import Sequence


def serialize(s):
    if isinstance(s, Tensor):
        return serialize_tensor(s)
    elif isinstance(s, FeatureDict):
        return serialize_featuredict(s)
    elif isinstance(s, Primitive):
        return serialize_primitive(s)
    else:
        raise TypeError("Unknown type", type(s))


def serialize_tensor(tensor):
    d = copy.deepcopy(tensor.__dict__)
    d["type"] = type(tensor).__name__
    if hasattr(tensor, 'dtype'):
        d["dtype"] = serialize(tensor.dtype)
    if hasattr(tensor, 'class_labels'):
        d["class_labels"] = serialize(tensor.class_labels)
    return d


def serialize_featuredict(fdict):
    d = {}
    d["type"] = "FeatureDict"
    d["items"] = {}
    for k, v in fdict.__dict__["dict_"].items():
        d["items"][k] = serialize(v)
    return d


def serialize_primitive(primitive):
    return str(primitive._dtype)
