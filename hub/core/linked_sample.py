from typing import Dict, Optional


class LinkedSample:
    def __init__(self, path: str, creds_key: Optional[str] = None):
        self.path = path
        self.creds_key = creds_key
