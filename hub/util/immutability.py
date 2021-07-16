from typing import Any, Sequence


def validate_can_be_parsed_as_immutable(item: Any, recursive: bool = True):
    # TODO: docstring

    if not isinstance(item, (int, float, str, list, tuple)):
        raise ValueError()  # TODO

    if recursive:
        if isinstance(item, (list, tuple)):
            for v in item:
                validate_can_be_parsed_as_immutable(v, recursive=True)


def recursively_parse_as_immutable(item: Any):
    # TODO: docstring

    validate_can_be_parsed_as_immutable(item, recursive=False)

    return item
