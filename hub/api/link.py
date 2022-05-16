from hub.core.linked_sample import LinkedSample
from typing import Optional, Dict


def link(
    path: str,
    creds_key: Optional[str] = None,
) -> LinkedSample:
    return LinkedSample(path, creds_key)
