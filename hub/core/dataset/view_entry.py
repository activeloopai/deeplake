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
        """Returns id of the view."""
        return self.info["id"]

    @property
    def query(self) -> Optional[str]:
        return self.info.get("query")

    @property
    def message(self) -> str:
        """Returns the message with which the view was saved."""
        return self.info.get("message", "")

    def __str__(self):
        return f"View(id='{self.id}', message='{self.message}', virtual={self.virtual})"

    __repr__ = __str__

    @property
    def virtual(self) -> bool:
        return self.info["virtual-datasource"]

    def load(self):
        "Loads the view and returns the `hub.Dataset`."
        ds = self._ds._sub_ds(".queries/" + (self.info.get("path") or self.info["id"]))
        if self.virtual:
            ds = ds._get_view(inherit_creds=not self._external)
        ds._is_view = True
        return ds

    def optimize(self, unlink=True):
        """Optimizes the view by copying the required data.

        Args:
            unlink (bool): Unlink linked tensors by copying data from the links to the view.

        Examples:
            ```
            # save view
            ds[:10].save_view(view_id="first_10")

            # optimize view
            ds.get_view("first_10").optimize()

            # load optimized view
            ds.load_view("first_10")
            ```
        """
        self.info = self._ds._optimize_saved_view(
            self.info["id"], external=self._external, unlink=unlink
        )
        return self

    def delete(self):
        """Deletes the view."""
        self._ds.delete_view(id=self.info["id"])
