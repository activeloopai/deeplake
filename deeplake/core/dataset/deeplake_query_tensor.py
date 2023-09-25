import deeplake.util.shape_interval as shape_interval
from deeplake.core import tensor
from typing import List, Union, Optional
from deeplake.core.index import Index
from deeplake.core.tensor import Any
import numpy as np
from deeplake.core.index import replace_ellipsis_with_slices
from deeplake.core.meta.tensor_meta import TensorMeta
from deeplake.util.exceptions import InvalidKeyTypeError, DynamicTensorNumpyError
from deeplake.util.pretty_print import summary_tensor
import json


class DeepLakeQueryTensor(tensor.Tensor):
    def __init__(
        self,
        deeplake_tensor,
        indra_tensor,
        index: Optional[Index] = None,
        is_iteration: bool = False,
    ):
        self.deeplake_tensor = deeplake_tensor
        self.indra_tensor = indra_tensor
        self.is_iteration = is_iteration

        self.key = (
            deeplake_tensor.key
            if hasattr(deeplake_tensor, "key")
            else indra_tensor.name
        )

        self.first_dim = None

        self._index = index or Index(self.indra_tensor.index)

    def __getattr__(self, key):
        try:
            return getattr(self.deeplake_tensor, key)
        except AttributeError:
            try:
                return getattr(self.indra_tensor, key)
            except AttributeError:
                raise AttributeError(
                    f"'{self.__class__}' object has no attribute '{key}'"
                )

    def __getitem__(
        self,
        item,
        is_iteration: bool = False,
    ):
        if not isinstance(item, (int, slice, list, tuple, type(Ellipsis), Index)):
            raise InvalidKeyTypeError(item)

        if isinstance(item, tuple) or item is Ellipsis:
            item = replace_ellipsis_with_slices(item, self.ndim)

        indra_tensor = self.indra_tensor[item]

        return DeepLakeQueryTensor(
            self.deeplake_tensor,
            indra_tensor,
            index=self.index[item],
            is_iteration=is_iteration,
        )

    def numpy(
        self, aslist=False, *args, **kwargs
    ) -> Union[np.ndarray, List[np.ndarray]]:
        r = self.indra_tensor.numpy(aslist=aslist)
        if aslist or isinstance(r, (np.ndarray, list)):
            return r
        else:
            try:
                return np.array(r)
            except ValueError:
                raise DynamicTensorNumpyError(self.name, self.index, "shape")

    def text(self, fetch_chunks: bool = False):
        """Return text data. Only applicable for tensors with 'text' base htype."""
        if self.ndim == 1:
            return self.indra_tensor.bytes().decode()
        return list(
            self.indra_tensor[i].bytes().decode() for i in range(len(self.indra_tensor))
        )

    def dict(self, fetch_chunks: bool = False):
        """Return json data. Only applicable for tensors with 'json' base htype."""
        if self.ndim == 1:
            return json.loads(self.indra_tensor.bytes().decode())
        return list(
            json.loads(self.indra_tensor[i].bytes().decode())
            for i in range(len(self.indra_tensor))
        )

    @property
    def dtype(self):
        return self.indra_tensor.dtype

    @property
    def htype(self):
        htype = self.indra_tensor.htype
        if self.indra_tensor.is_sequence:
            htype = f"sequence[{htype}]"
        if self.deeplake_tensor.is_link:
            htype = f"link[{htype}]"
        return htype

    @htype.setter
    def htype(self, value):
        raise NotImplementedError("htype of a virtual tensor cannot be set.")

    @property
    def sample_compression(self):
        return self.indra_tensor.sample_compression

    @property
    def chunk_compression(self):
        return None

    @property
    def num_samples(self):
        return len(self.indra_tensor)

    def can_convert_to_numpy(self):
        return None not in self.shape

    @property
    def max_shape(self):
        return self.indra_tensor.max_shape

    @property
    def min_shape(self):
        return self.indra_tensor.min_shape

    @property
    def chunk_engine(self):
        raise NotImplementedError("Virtual tensor does not have chunk engine.")

    @property
    def shape(self):
        if (
            not self.indra_tensor.is_sequence
            and len(self.indra_tensor) == 1
            and self.index.values[0].subscriptable()
        ):
            return (len(self.indra_tensor), *self.indra_tensor.shape)
        else:
            return self.indra_tensor.shape

    @property
    def index(self):
        if self._index is not None:
            return self._index
        return Index(self.indra_tensor.indexes)

    @property
    def shape_interval(self):
        return shape_interval.ShapeInterval(self.min_shape, self.max_shape)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def meta(self):
        """Metadata of the tensor."""
        if self.deeplake_tensor is None:
            return TensorMeta(
                htype=self.indra_tensor.htype,
                dtype=self.indra_tensor.dtype,
                sample_compression=self.indra_tensor.sample_compression,
                chunk_compression=None,
                is_sequence=self.indra_tensor.is_sequence,
                is_link=False,
            )
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
        return self.meta.htype

    def __len__(self):
        return len(self.indra_tensor)

    def summary(self):
        """Prints a summary of the tensor."""
        pretty_print = summary_tensor(self)

        print(self)
        print(pretty_print)
