from typing import Optional
from deeplake.constants import ALL_CLOUD_PREFIXES
import numpy as np


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
