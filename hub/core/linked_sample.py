from typing import Dict, Optional


class LinkedSample:
    def __init__(self, path: str, creds: Optional[Dict] = None):
        self.path = path
        self.creds = creds
