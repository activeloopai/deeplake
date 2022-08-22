from typing import Optional
from hub.util.link import convert_creds_key


class LinkedSample:
    def __init__(self, path: str, creds_key: Optional[str] = None):
        self.path = path
        self.creds_key = convert_creds_key(creds_key, path)
