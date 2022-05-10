from typing import Dict, Optional, Any


class ViewEntry:
    def __init__(self, info: Dict, dataset):
        self.info = info
        self._ds = dataset

    def __getitem__(self, key: str):
        return self.info[key]

    def get(self, key: str, default: Optional[Any] = None):
        return self.info.get(key)

    @property
    def id(self) -> str:
        return self.info["id"]

    @property
    def message(self) -> str:
        return self.info.get("message", "")

    def __str__(self):
        return f"View(id='{self.id}', message='{self.message}', virtual={self.virtual})"

    @property
    def virtual(self) -> bool:
        return self.ifo["virtual-datasource"]

    def load(self):
        return self._ds._get_sub_ds(".queries/" + self.info[id])
