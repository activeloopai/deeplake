from hub.util.immutability import PARSABLE_TYPES
import json
from typing import Any


def validate_is_jsonable(key: str, item: Any):
    # TODO: docstring

    try:
        json.dumps(item)
    except Exception:
        raise ValueError(
            f"Item for key='{key}' is not JSON serializable. Allowed types: {str(PARSABLE_TYPES)} Got: type={type(item)}, item={item}"
        )
