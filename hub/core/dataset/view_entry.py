from typing import Dict, Optional, Any


class ViewEntry:
    def __init__(self, info: Dict):
        self.info = info

    def __getitem__(self, key: str):
        return self.info[key]

    def get(self, key: str, default: Optional[Any] = None):
        return self.info.get(key)

    @property
    def id(self) -> str:
        return self["id"]

    @property
    def message(self) -> str:
        return self.get("message", "")

    def __str__(self):
        return f"View(id='{self.id}', message='{self.message}')"
