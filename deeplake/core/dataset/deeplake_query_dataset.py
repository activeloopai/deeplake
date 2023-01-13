import posixpath
import warnings
from typing import List, Tuple, Union, Dict, Optional, Callable

# from deeplake.core.dataset.dataset import DeepLakeQueryDataset
from deeplake.util.exceptions import (
    InvalidOperationError,
    TensorDoesNotExistError,
    InvalidKeyTypeError,
    MemoryDatasetCanNotBePickledError,
)
import deeplake
from deeplake.core.index import Index

try:
    from indra.pytorch.loader import Loader
    from indra.pytorch.common import collate_fn as default_collate

    _INDRA_INSTALLED = True
except ImportError:
    _INDRA_INSTALLED = False
from deeplake.core.dataset.deeplake_query_tensor import DeepLakeQueryTensor
from deeplake.util.iteration_warning import (
    check_if_iteration,
)

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
        ds = self._ds._sub_ds(
            ".queries/" + (self.info.get("path") or self.info["id"]),
            lock=False,
            verbose=False,
        )
        sub_ds_path = ds.path
        if self.virtual:
            ds = ds._get_view(inherit_creds=not self._external)
        ds._view_entry = self
        if verbose:
            log_visualizer_link(sub_ds_path, source_ds_url=self.info["source-dataset"])

        query_str = self.info.get("query")
        indra_ds = ds.query(query_str)

        view = DeepLakeQueryDataset(deeplake_ds=ds, indra_ds=indra_ds)
        return view
