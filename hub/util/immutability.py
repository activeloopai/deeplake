from typing import Any


PARSABLE_TYPES = (int, float, str, list, tuple)
PARSABLE_SEQUENCE_TYPES = (list, tuple)


def validate_can_be_parsed_as_immutable(item: Any, recursive: bool = True, key=None):
    # TODO: docstring

    if not isinstance(item, PARSABLE_TYPES):
        raise ValueError()  # TODO (mention `key`)

    if recursive:
        if isinstance(item, PARSABLE_SEQUENCE_TYPES):
            for v in item:
                validate_can_be_parsed_as_immutable(v, recursive=True, key=key)


def recursively_parse_as_immutable(item: Any):
    # TODO: docstring

    return item
