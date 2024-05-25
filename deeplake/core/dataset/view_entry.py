from typing import Dict, Optional, Any, List
from deeplake.util.logging import log_visualizer_link


class ViewEntry:
    """Represents a view saved inside a dataset."""

    def __init__(
        self, info: Dict, dataset, source_dataset=None, external: bool = False
    ):
        self.info = info
        self._ds = dataset
        self._src_ds = source_dataset if external else dataset
        self._external = external

    def __getitem__(self, key: str):
        return self.info[key]

    def get(self, key: str, default: Optional[Any] = None):
        return self.info.get(key, default)

    @property
    def id(self) -> str:
        """Returns id of the view."""
        return self.info["id"].split("]")[-1]

    @property
    def query(self) -> Optional[str]:
        return self.info.get("query")

    @property
    def tql_query(self) -> Optional[str]:
        return self.info.get("tql_query")

    @property
    def message(self) -> str:
        """Returns the message with which the view was saved."""
        return self.info.get("message", "")

    @property
    def commit_id(self) -> str:
        return self.info["source-dataset-version"]

    def __str__(self):
        return f"View(id='{self.id}', message='{self.message}', virtual={self.virtual}, commit_id={self.commit_id}, query='{self.query}, tql_query='{self.tql_query}')"

    __repr__ = __str__

    @property
    def virtual(self) -> bool:
        return self.info["virtual-datasource"]

    def load(self, verbose=True):
        """Loads the view and returns the :class:`~deeplake.core.dataset.Dataset`.

        Args:
            verbose (bool): If ``True``, logs will be printed. Defaults to ``True``.

        Returns:
            Dataset: Loaded dataset view.
        """
        if self.commit_id != self._ds.commit_id:
            print(f"Loading view from commit id {self.commit_id}.")

        ds = self._ds._sub_ds(
            ".queries/" + (self.info.get("path") or self.info["id"]),
            lock=False,
            verbose=False,
            token=self._src_ds.token,
            read_only=True,
        )

        sub_ds_path = ds.path
        if self.virtual:
            ds = ds._get_view(inherit_creds=not self._external)

        if self.virtual and not self.tql_query is None:
            query_str = self.tql_query
            ds = ds.query(query_str)

        ds._view_entry = self
        if verbose:
            log_visualizer_link(sub_ds_path, source_ds_url=self.info["source-dataset"])
        return ds

    def optimize(
        self,
        tensors: Optional[List[str]] = None,
        unlink=True,
        num_workers=0,
        scheduler="threaded",
        progressbar=True,
    ):
        """Optimizes the dataset view by copying and rechunking the required data. This is necessary to achieve fast streaming
        speeds when training models using the dataset view. The optimization process will take some time, depending on
        the size of the data.

        Example:

            >>> # save view
            >>> ds[:10].save_view(id="first_10")
            >>> # optimize view
            >>> ds.get_view("first_10").optimize()
            >>> # load optimized view
            >>> ds.load_view("first_10")

        Args:
            tensors (List[str]): Tensors required in the optimized view. By default all tensors are copied.
            unlink (bool): - If ``True``, this unlinks linked tensors (if any) by copying data from the links to the view.
                    - This does not apply to linked videos. Set ``deeplake.constants._UNLINK_VIDEOS`` to ``True`` to change this behavior.
            num_workers (int): Number of workers to be used for the optimization process. Defaults to 0.
            scheduler (str): The scheduler to be used for optimization. Supported values include: 'serial', 'threaded', and 'processed'.
                Only applicable if ``optimize=True``. Defaults to 'threaded'.
            progressbar (bool): Whether to display a progressbar.

        Returns:
            :class:`ViewEntry`

        Raises:
            Exception: When query view cannot be optimized.

        """
        self.info = self._ds._optimize_saved_view(
            self.info["id"],
            tensors=tensors,
            external=self._external,
            unlink=unlink,
            num_workers=num_workers,
            scheduler=scheduler,
            progressbar=progressbar,
        )
        return self

    @property
    def source_dataset_path(self) -> str:
        return self.info["source-dataset"]

    def delete(self):
        """Deletes the view."""
        self._ds.delete_view(id=self.info["id"])
