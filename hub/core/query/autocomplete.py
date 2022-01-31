import hub
from tokenize import tokenize, TokenError
from io import BytesIO
from typing import Any, Dict, List


def _token_obj_to_dict(token):
    if token.end[0] != 1:
        raise NotImplementedError("Only single line queries are supported yet.")
    return {
        "string": token.string,
        "start": token.start[1],
        "end": token.end[1],
        #  Note: type will be filled in by _tokenize()
    }


_TENSOR_PROPERTIES = set(
    (
        "min",
        "max",
        "mean",
        "shape",
        "size",
    )
)

_TENSOR_METHODS = set(("contains",))


_PYTHON_KEYWORDS = set(
    (
        "in",
        "is",
        "and",
        "or",
        "not",
        "if",
        "else",
        "for",
    )
)

_PYTHON_CONSTANTS = set(
    (
        "True",
        "False",
        "None",
    )
)


def _tokenize(s: str):
    return list(tokenize(BytesIO(s.encode("utf-8")).readline))[1:-2]


def _parse(s: str, ds: hub.Dataset) -> List[dict]:
    pytokens = _tokenize(s)
    tensors = ds._ungrouped_tensors
    groups = set(ds._groups_filtered)
    hubtokens: List[Dict[str, Any]] = []
    group_in_progress = None
    for i, t in enumerate(pytokens):
        ht = _token_obj_to_dict(t)
        ht["type"] = "UNKNOWN"
        ts = t.string
        if t.type == 1:  # (NAME)
            if group_in_progress:
                if ts in group_in_progress._ungrouped_tensors:
                    ht["type"] = "TENSOR"
                    group_in_progress = None
                elif ts in group_in_progress._groups_filtered:
                    ht["type"] = "GROUP"
                    group_in_progress = group_in_progress[ts]
                else:
                    group_in_progress = None
            if ts in _PYTHON_KEYWORDS:
                ht["type"] = "KEYWORD"
            elif ts in _PYTHON_CONSTANTS:
                ht["type"] = "CONSTANT"
            elif ts in tensors:
                ht["type"] = "TENSOR"
            elif ts in groups:
                ht["type"] = "GROUP"
                group_in_progress = ds[ts]
            elif ts in _TENSOR_PROPERTIES:
                if (
                    i >= 2
                    and hubtokens[-1]["string"] == "."
                    and hubtokens[-2]["type"] == "TENSOR"
                ):
                    ht["type"] = "PROPERTY"
                else:
                    pass  # Syntax error
            elif ts in _TENSOR_METHODS:
                if (
                    i >= 2
                    and hubtokens[-1]["string"] == "."
                    and hubtokens[-2]["type"] == "TENSOR"
                ):
                    ht["type"] = "METHOD"
                else:
                    pass  # Syntax error
        elif t.type == 2:
            ht["type"] = "NUMBER"
            group_in_progress = None
        elif t.type == 3:
            ht["type"] = "STRING"
            group_in_progress = None
        elif t.type == 53:  # OP
            if ts != ".":
                group_in_progress = None
            elif i >= 2 and hubtokens[-1]["string"] == ".":
                group_in_progress = None  # Syntax error
            ht["type"] = "OP"
        hubtokens.append(ht)
    return hubtokens


def _parse_no_fail(s: str, ds: hub.dataset):
    try:
        return _parse(s, ds)
    except TokenError:
        return _parse(s[:-1], ds)


def _initial_suggestions(ds):
    return [{"string": k, "type": "TENSOR"} for k in ds._ungrouped_tensors] + [
        {"string": k, "type": "GROUP"} for k in ds._groups_filtered
    ]


def _tensor_suggestions(ds, tensor):
    suggestions = []
    for k in _TENSOR_PROPERTIES:
        suggestions.append({"string": k, "type": "PROPERTY"})
    for k in _TENSOR_METHODS:
        suggestions.append({"string": k, "type": "METHOD"})
    return suggestions


def _const_suggestions():
    return [{"string": str(v), "type": "CONSTANT"} for v in (True, False, None)]


def _group_suggestions(ds, group):
    return [{"string": k, "type": "TENSOR"} for k in group._ungrouped_tensors] + [
        {"string": k, "type": "GROUP"} for k in group._groups_filtered
    ]


def _parse_last_tensor(tokens, ds):
    tensor = []
    n = len(tokens) - 1
    if tokens[n]["type"] in ("UNKNOWN", "OP"):
        n -= 1
    for i in range(n, -1, -1):
        if tokens[i]["string"] == ".":
            continue
        elif tokens[i]["type"] in ("TENSOR", "GROUP"):
            tensor.append(tokens[i]["string"])
        else:
            break
    ret = ds
    for i in range(len(tensor) - 1, -1, -1):
        ret = ret[tensor[i]]
    return ret


def _filter(suggestions: List[dict], string: str):
    return list(
        filter(lambda x: x["string"].lower().startswith(string.lower()), suggestions)
    )


def _autocomplete_response(
    suggestions: List[dict], tokens: List[dict], replace: str = ""
):
    return {
        "suggestions": suggestions,
        "tokens": tokens,
        "replace": replace,
    }


def _prefix_suggestions(suggestions: List[dict], prefix: str) -> List[dict]:
    return list(
        map(lambda s: {"string": prefix + s["string"], "type": s["type"]}, suggestions)
    )


def _op_suggestions():
    ops = [
        "==",
        ">",
        "<",
        ">=",
        "<=",
        "!=",
    ]
    return [{"string": op, "type": "OP"} for op in ops]


def autocomplete(s: str, ds: hub.dataset) -> dict:
    if not s.strip():
        return _autocomplete_response(_initial_suggestions(ds), [])
    try:
        tokens = _parse(s, ds)
    except TokenError:
        return _autocomplete_response([], _parse_no_fail(s[:-1], ds))

    last_token = tokens[-1]
    last_type = last_token["type"]
    last_string = last_token["string"]

    if last_type == "UNKNOWN":
        if s[-1] == " ":
            return _autocomplete_response([], tokens)
        if len(tokens) == 1:
            suggestions = _filter(_initial_suggestions(ds), last_string)
            return _autocomplete_response(suggestions, tokens, last_string)
        if len(tokens) >= 3 and tokens[-2]["string"] == ".":
            prev_token = tokens[-3]
            prev_type = prev_token["type"]
            if prev_type == "TENSOR":
                tensor = _parse_last_tensor(tokens, ds)
                suggestions = _filter(_tensor_suggestions(ds, tensor), last_string)
                return _autocomplete_response(suggestions, tokens, last_string)
            elif prev_type == "GROUP":
                group = _parse_last_tensor(tokens, ds)
                suggestions = _filter(_group_suggestions(ds, group), last_string)
                return _autocomplete_response(suggestions, tokens, last_string)
            else:
                return _autocomplete_response([], tokens)
        suggestions = _filter(
            _initial_suggestions(ds) + _const_suggestions(), last_string
        )
        return _autocomplete_response(suggestions, tokens, last_string)
    elif last_type == "OP":
        if last_string == ".":
            if len(tokens) == 1:
                return _autocomplete_response([], tokens)
            prev_token = tokens[-2]
            prev_type = prev_token["type"]
            if prev_type == "UNKNOWN":
                return _autocomplete_response([], tokens)
            elif prev_type == "TENSOR":
                tensor = _parse_last_tensor(tokens, ds)
                return _autocomplete_response(_tensor_suggestions(ds, tensor), tokens)
            elif prev_type == "GROUP":
                group = _parse_last_tensor(tokens, ds)
                return _autocomplete_response(_tensor_suggestions(ds, group), tokens)
            else:
                return _autocomplete_response([], tokens)
        else:
            suggestions = _initial_suggestions(ds) + _const_suggestions()
            return _autocomplete_response(suggestions, tokens, last_string)
    elif last_type == "KEYWORD":
        suggestions = _initial_suggestions(ds) + _const_suggestions()
        return _autocomplete_response(
            _prefix_suggestions(suggestions, "" if s[-1] == " " else " "),
            tokens,
            last_string,
        )
    elif last_type == "TENSOR":
        tensor = _parse_last_tensor(tokens, ds)
        suggestions = _prefix_suggestions(
            _tensor_suggestions(ds, tensor), "."
        ) + _prefix_suggestions(_op_suggestions(), "" if s[-1] == " " else " ")
        return _autocomplete_response(suggestions, tokens)
    elif last_type == "GROUP":
        group = _parse_last_tensor(tokens, ds)
        suggestions = _prefix_suggestions(_group_suggestions(ds, group), ".")
        return _autocomplete_response(suggestions, tokens)
    elif last_type == "METHOD":
        return _autocomplete_response([{"string": "(", "type": "OP"}], tokens)
    elif last_type == "PROPERTY":
        suggestions = _prefix_suggestions(
            _op_suggestions(), "" if s[-1] == " " else " "
        )
        return _autocomplete_response(suggestions, tokens)
    else:
        return _autocomplete_response([], tokens)
