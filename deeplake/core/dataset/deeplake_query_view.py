from typing import Dict

import deeplake as dp
import deeplake.core.dataset.deeplake_query_dataset as deeplake_query_dataset
from deeplake.core.dataset.view_entry import ViewEntry
from deeplake.util.logging import log_visualizer_link


class NonlinearQueryView(ViewEntry):
    def __init__(
        self, info: Dict, dataset, source_dataset=None, external: bool = False
    ):
        self.info = info
        self._ds = dataset
        self._src_ds = source_dataset if external else dataset
        self._external = external

    def load(self, verbose=True):
        """Loads the view and returns the :class:`~deeplake.core.dataset.Dataset`.

        Args:
            verbose (bool): If ``True``, logs will be printed. Defaults to ``True``.

        Returns:
            Dataset: Loaded dataset view.
        """
        sub_ds = self._ds._sub_ds(
            ".queries/" + (self.info.get("path") or self.info["id"]),
            lock=False,
            verbose=False,
        )
        sub_ds_path = sub_ds.path.split("/.queries/")[0]
        ds = dp.load(sub_ds_path)
        ds._view_entry = self
        if verbose:
            log_visualizer_link(sub_ds_path, source_ds_url=self.info["source-dataset"])

        query_str = self.info.get("query")
        view = ds.query(query_str)
        return view
