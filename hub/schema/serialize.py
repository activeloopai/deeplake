"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import copy

from hub.schema.features import Primitive, Tensor, SchemaDict


def serialize(input):
    "Converts the input into a serializable format"
    if isinstance(input, Tensor):
        return serialize_tensor(input)
    elif isinstance(input, SchemaDict):
        return serialize_SchemaDict(input)
    elif isinstance(input, Primitive):
        return serialize_primitive(input)
    else:
        raise TypeError("Unknown type", type(input))


def serialize_tensor(tensor):
    "Converts Tensor and its derivatives into a serializable format"
    d = copy.deepcopy(tensor.__dict__)
    d["type"] = type(tensor).__name__
    if hasattr(tensor, "dtype"):
        d["dtype"] = serialize(tensor.dtype)
    if hasattr(tensor, "class_labels"):
        d["class_labels"] = serialize(tensor.class_labels)
    return d


def serialize_SchemaDict(fdict):
    "Converts SchemaDict into a serializable format"
    d = {}
    d["type"] = "SchemaDict"
    d["items"] = {}
    for k, v in fdict.__dict__["dict_"].items():
        d["items"][k] = serialize(v)
    return d


def serialize_primitive(primitive):
    "Converts Primitive into a serializable format"
    return str(primitive._dtype)
