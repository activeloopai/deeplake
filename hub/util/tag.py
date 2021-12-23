from hub.util.exceptions import InvalidHubPathException
from typing import Tuple


def process_hub_path(path: str) -> Tuple[str, str, str, str]:
    """Checks whether path is a valid hub path."""
    # Allowed formats:
    # hub://org/ds
    # hub://org/ds/.queries/hash
    # hub://org/queries/hash
    # hub//org/queries/.queries/hash

    tag = path[6:]
    s = tag.split("/")
    if len(s) == 2:
        if s[1] == "queries":  # Attempting to open queries ds root
            raise InvalidHubPathException(path)
        return (path, *s, "")
    elif len(s) == 3:
        if s[1] != "queries":
            raise InvalidHubPathException(path)
        return (f"hub://{s[0]}/queries/.queries/{s[2]}", *s[:2], f".queries/{s[2]}")
    elif len(s) == 4:
        if s[2] != ".queries":
            raise InvalidHubPathException(path)
        return (path, *s[:2], f".queries/{s[3]}")
    else:
        raise InvalidHubPathException(path)
