from deeplake.util.exceptions import InvalidHubPathException
from typing import Tuple
import deeplake


def process_hub_path(path: str) -> Tuple[str, str, str, str]:
    """Checks whether path is a valid Deep Lake cloud path."""
    # Allowed formats:
    # hub://org/ds
    # hub://org/ds/.queries/hash
    # hub://org/queries/hash
    # hub://org/queries/.queries/hash
    # hub://org/ds/sub_ds1/sub_ds2/sub_ds3/..../sub_ds{n}  # Only for internal usage.

    tag = path[6:]
    s = tag.split("/")

    if len(s) < 2:
        raise InvalidHubPathException(path)

    path = f"hub://{s[0]}/{s[1]}"

    if len(s) == 3 and s[1] == "queries" and not s[2].startswith("."):
        # Special case: expand hub://username/queries/hash to hub://username/queries/.queries/hash
        subdir = f".queries/{s[2]}"
    else:
        subdir = "/".join(s[2:])
        if len(subdir) and len(s) > 2:
            if (
                not (len(s) == 4 and s[2] == ".queries")
                and not deeplake.constants._ENABLE_HUB_SUB_DATASETS
            ):
                raise InvalidHubPathException(path)
    return (path, *s[:2], subdir)  # type: ignore
