from typing import Dict, Optional, Any


class ViewEntry:
    """Represents a view saved inside a dataset."""

    def __init__(self, info: Dict, dataset, external: bool = False):
        self.info = info
        self._ds = dataset
        self._external = external

    def __getitem__(self, key: str):
        return self.info[key]

    def get(self, key: str, default: Optional[Any] = None):
        return self.info.get(key, default)

    @property
    def id(self) -> str:
        return self.info["id"]

    @property
    def query(self) -> Optional[str]:
        return self.info.get("query")

    @property
    def message(self) -> str:
        return self.info.get("message", "")

    def __str__(self):
        return f"View(id='{self.id}', message='{self.message}', virtual={self.virtual})"

    @property
    def virtual(self) -> bool:
        return self.info["virtual-datasource"]

    def load(self):
        ds = self._ds._sub_ds(".queries/" + (self.info.get("path") or self.info["id"]))
        if self.virtual:
            ds = ds._get_view(inherit_creds=not self._external)
        return ds

    def optimize(self, unlink=True):
        self.info = self._ds._optimize_saved_view(
            self.info["id"], external=self._external, unlink=unlink
        )

    def delete(self):
        self._ds.delete_view(id=self.info["id"])
