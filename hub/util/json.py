import json
from typing import Any


def validate_is_jsonable(key: str, item: Any):
    """Validates if `item` can be parsed with the `json` package.

    Args:
        key (str): Key for the item. This is printed in the exception.
        item (Any): `item` that should be parsable with the `json` package.

    Raises:
        ValueError: If `item` is not `json` parsable.
    """

    try:
        json.dumps(item)
    except Exception:
        raise ValueError(
            f"Item for key='{key}' is not JSON serializable. Got: type={type(item)}, item={item}"
        )
