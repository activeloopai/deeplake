from hub.core.linked_sample import LinkedSample
from typing import Optional, Dict


def link(
    path: str,
    creds: Optional[Dict] = None,
) -> LinkedSample:
    return LinkedSample(path, creds)
