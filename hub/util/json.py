from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from numpy import ndarray
import json
import base64
from hub.core.sample import Sample  # type: ignore

Schema = Any


scalars = ["int", "float", "bool", "str", "list", "dict", "ndarray", "Sample"]
types = ["Any", "Dict", "List", "Optional", "Union"]


def _norm_type(typ: str):
    typ = typ.replace("typing.", "")
    replacements = {
        "numpy.ndarray": "ndarray",
        "np.ndarray": "ndarray",
        "hub.core.sample.Sample": "Sample",
        "hub.Sample": "Sample",
    }
    return replacements.get(typ, typ)


def _parse_schema(schema: Union[str, Schema]) -> Tuple[str, List[str]]:
    if getattr(schema, "__module__", None) == "typing":
        schema = str(schema)
        validate = False
    else:
        validate = True

    if schema in scalars:
        return schema, []

    if "[" not in schema:
        return _norm_type(schema), []

    typ, param_string = schema.split("[", 1)
    typ = _norm_type(typ)
    assert param_string[-1] == "]"
    params = []
    buff = ""
    level = 0
    for c in param_string:
        if c == "[":
            level += 1
            buff += c
        elif c == "]":
            if level == 0:
                if buff:
                    params.append(buff)
                if validate:
                    _validate_schema(typ, params)
                return typ, params
            else:
                buff += c
                level -= 1
        elif c == ",":
            if level == 0:
                params.append(buff)
                buff = ""
            else:
                buff += c
        elif c == " ":
            continue
        else:
            buff += c
    raise InvalidJsonSchemaException()


class InvalidJsonSchemaException(Exception):
    pass


class ArgumentMismatchException(InvalidJsonSchemaException):
    def __init__(self, typ: str, actual: int, expected: int, exact: bool = False):
        assert actual != expected
        gt = actual > expected
        super(ArgumentMismatchException, self).__init__(
            f"Too {'many' if gt else 'few'} parameters for {typ};"
            + f" actual {actual},expected {'exatcly' if exact else ('at most' if gt else 'at least')} {expected}."
        )


def _validate_schema(typ: str, params: List[str]) -> Tuple[str, List[str]]:
    if typ in scalars:
        return typ, params

    if typ not in types:
        raise InvalidJsonSchemaException(f"Unsupported type: {typ}")

    def _err(expected_num_params: int, exact: bool = False):
        raise ArgumentMismatchException(typ, len(params), expected_num_params, exact)

    if typ == "Any":
        if params:
            _err(0)
    elif typ == "Optional":
        if len(params) > 1:
            _err(1)
    elif typ == "Union":
        if len(params) == 0:
            _err(1)
    elif typ == "List":
        if len(params) > 1:
            _err(1)
    elif typ == "Dict":
        if len(params) not in (0, 2):
            _err(2, True)
    return typ, params


def _validate_any(obj: Any, params: List[str]):
    assert not params
    return True


def _validate_union(obj: Any, params: List[str]):
    for schema in params:
        if _validate_object(obj, schema):
            return True
    return False


def _validate_optional(obj: Any, params: List[str]) -> bool:
    assert len(params) <= 1
    if obj is None:
        return True
    if params:
        return _validate_object(obj, params[0])
    return True


def _validate_list(obj: Any, params: List[str]) -> bool:
    assert len(params) <= 1
    if not isinstance(obj, (list, tuple)):
        return False
    if params:
        for item in obj:
            if not _validate_object(item, params[0]):
                return False
    return True


def _validate_dict(obj: Any, params: List[str]) -> bool:
    assert len(params) in (0, 2)
    if not isinstance(obj, dict):
        return False
    if params:
        assert params[0] in (
            "str",
            "Any",
        ), "Only string keys are allowed for json dicts."
        for v in obj.values():
            if not _validate_object(v, params[1]):
                return False
    return True


def _validate_nonetype(obj: Any, params: List[str]) -> bool:
    assert not params
    return obj is None


def _validate_object(obj: Any, schema: Union[str, Schema]) -> bool:
    typ, params = _parse_schema(schema)
    if typ in scalars:
        return isinstance(obj, eval(typ))
    return globals()[f"_validate_{typ.lower()}"](obj, params)


class JsonValidationError(Exception):
    pass


def validate_json_object(obj: Any, schema: Union[str, Schema]) -> None:
    if obj and not _validate_object(obj, schema):
        raise JsonValidationError()


def validate_json_schema(schema: str):
    _parse_schema(schema)


class HubJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ndarray):
            return {
                "_hub_custom_type": "ndarray",
                "data": base64.b64encode(obj.tobytes()).decode(),
                "shape": obj.shape,
                "dtype": obj.dtype.name,
            }
        elif isinstance(obj, Sample):
            if obj.compression:
                return {
                    "_hub_custom_type": "Sample",
                    "data": base64.b64encode(obj.buffer).decode(),
                    "compression": obj.compression,
                }
            else:
                return self.default(obj.array)
        return obj


class HubJsonDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        hub_custom_type = obj.get("_hub_custom_type")
        if hub_custom_type == "ndarray":
            return np.frombuffer(
                base64.b64decode(obj["data"]), dtype=obj["dtype"]
            ).reshape(obj["shape"])
        elif hub_custom_type == "Sample":
            return Sample(
                buffer=base64.b64decode(obj["data"]), compression=obj["compression"]
            )
        return obj
