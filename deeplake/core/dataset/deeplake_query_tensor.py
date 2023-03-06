import deeplake.util.shape_interval as shape_interval
from deeplake.core import tensor
from typing import List, Union, Optional
from deeplake.core.index import Index
from deeplake.core.tensor import Any
import numpy as np
import itertools
from deeplake.core.index import replace_ellipsis_with_slices
from deeplake.util.exceptions import DynamicTensorNumpyError, InvalidKeyTypeError
from deeplake.util.pretty_print import summary_tensor


class DeepLakeQueryTensor(tensor.Tensor):
    def __init__(
        self,
        deeplake_tensor,
        indra_tensor,
        is_iteration: bool = False,
        key: Optional[str] = None,
    ):
        self.deeplake_tensor = deeplake_tensor
        self.indra_tensor = indra_tensor
        self.is_iteration = is_iteration
        self.set_deeplake_tensor_variables()

        if key:
            self.key = key

        self.first_dim = None

    def set_deeplake_tensor_variables(self):
        attrs = [
            "key",
            "dataset",
            "storage",
            "version_state",
            "link_creds",
            "_skip_next_setitem",
            "_indexing_history",
        ]

        for k in attrs:
            if hasattr(self.deeplake_tensor, k):
                setattr(self, k, getattr(self.deeplake_tensor, k))

        commit_id = self.deeplake_tensor.version_state["commit_id"]

    def __getitem__(
        self,
        item,
        is_iteration: bool = False,
    ):
        if not isinstance(item, (int, slice, list, tuple, type(Ellipsis), Index)):
            raise InvalidKeyTypeError(item)

        if isinstance(item, tuple) or item is Ellipsis:
            item = replace_ellipsis_with_slices(item, self.ndim)

        key = None
        if hasattr(self, "key"):
            key = self.key

        indra_tensor = self.indra_tensor[item]

        return DeepLakeQueryTensor(
            self.deeplake_tensor[item],
            indra_tensor,
            is_iteration=is_iteration,
            key=key,
        )

    def numpy(
        self, aslist=False, *args, **kwargs
    ) -> Union[np.ndarray, List[np.ndarray]]:
        return self.indra_tensor.numpy(aslist=aslist)

    def data(self, aslist: bool = False, fetch_chunks: bool = False) -> Any:
        return self.indra_tensor.bytes()

    @property
    def num_samples(self):
        return len(self.indra_tensor)

    def can_convert_to_numpy(self):
        if None in self.shape:
            return False
        return True

    @property
    def max_shape(self):
        return self.indra_tensor.max_shape

    @property
    def min_shape(self):
        return self.indra_tensor.min_shape

    @property
    def shape(self):
        if self.max_shape != self.min_shape:
            shape = []
            for i, dim_len in enumerate(self.max_shape):
                if dim_len == self.min_shape[i]:
                    shape.append(self.min_shape[i])
                else:
                    shape.append(None)
        return shape
    
    @property
    def index(self):
        return self.indra_tensor.indexes

    @property
    def shape_interval(self):
        return shape_interval.ShapeInterval(self.min_shape, self.max_shape)

    @property
    def ndim(self):
        return len(self.max_shape)

    @property
    def htype(self):
        """Htype of the tensor."""
        htype = self.indra_tensor.htype

    @property
    def meta(self):
        """Metadata of the tensor."""
        return self.deeplake_tensor.chunk_engine.tensor_meta

    @property
    def base_htype(self):
        """Base htype of the tensor.

        Example:

            >>> ds.create_tensor("video_seq", htype="sequence[video]", sample_compression="mp4")
            >>> ds.video_seq.htype
            sequence[video]
            >>> ds.video_seq.base_htype
            video
        """
        return self.deeplake_tensor.meta.htype

    def __len__(self):
        return len(self.indra_tensor)

    def summary(self):
        """Prints a summary of the tensor."""
        pretty_print = summary_tensor(self)

        print(self)
        print(pretty_print)
