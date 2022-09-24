from typing import Optional
<<<<<<< HEAD:hub/core/linked_sample.py
from hub.constants import ALL_CLOUD_PREFIXES
import numpy as np
=======
from deeplake.constants import ALL_CLOUD_PREFIXES
>>>>>>> f13af75bb31f6b78c232d9a0ac9f7ce32ccb5922:deeplake/core/linked_sample.py


def convert_creds_key(creds_key: Optional[str], path: str):
    if creds_key is None and path.startswith(ALL_CLOUD_PREFIXES):
        creds_key = "ENV"
    elif creds_key == "ENV" and not path.startswith(ALL_CLOUD_PREFIXES):
        creds_key = None
    return creds_key


class LinkedSample:
    """Represents a sample that is initialized using external links. See :meth:`deeplake.link`."""

    def __init__(self, path: str, creds_key: Optional[str] = None):
        self.path = path
        self.creds_key = convert_creds_key(creds_key, path)

    def dtype(self) -> str:
        return np.array("").dtype.name
