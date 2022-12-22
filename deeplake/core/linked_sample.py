from typing import Optional
import numpy as np

from deeplake.util.creds import convert_creds_key


class LinkedSample:
    """Represents a sample that is initialized using external links. See :meth:`deeplake.link`."""

    def __init__(self, path: str, creds_key: Optional[str] = None):
        self.path = path
        self.creds_key = convert_creds_key(creds_key, path)

    def dtype(self) -> str:
        return np.array("").dtype.name
