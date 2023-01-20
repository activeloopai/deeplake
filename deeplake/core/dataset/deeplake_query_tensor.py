import deeplake.util.shape_interval as shape_interval
from deeplake.core import tensor
from typing import List, Union
from deeplake.core.index import Index
import numpy as np
from deeplake.core.index import replace_ellipsis_with_slices
from deeplake.util.exceptions import DynamicTensorNumpyError, InvalidKeyTypeError


try:
    from indra.pytorch.loader import Loader
    from indra.pytorch.common import collate_fn as default_collate

    _INDRA_INSTALLED = True
except ImportError:
    _INDRA_INSTALLED = False


class DeepLakeQueryTensor(tensor.Tensor):
    def __init__(
        self,
        deeplake_tensor,
        indra_tensors,
        index=None,
        is_iteration: bool = False,
        key: str = None,
    ):
        self.deeplake_tensor = deeplake_tensor
        self.indra_tensors = indra_tensors
        self.is_iteration = is_iteration
        self.set_deeplake_tensor_variables()

        if index:
            self.index = index

        if key:
            self.key = key

        self.first_dim = None

    def set_deeplake_tensor_variables(self):
        attrs = [
            "key",
            "dataset",
            "storage",
            "index",
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

        DeeplakeIndraTensor = DeepLakeQueryTensor
        if self.index:
            DeeplakeIndraTensor = cast_index_to_class(self.index[item])

        return DeeplakeIndraTensor(
            self.deeplake_tensor,
            self.indra_tensors,
            index=self.index[item],
            is_iteration=is_iteration,
            key=key,
        )

    def numpy_aslist(self, aslist):
        tensors_list = self.indra_tensors[:]
        idx = self.index.values[0].value
        if idx:
            tensors_list = self.indra_tensors[idx]

        if self.min_shape != self.max_shape and self.aslist == False:
            raise DynamicTensorNumpyError(self.key, self.index, "shape")

        return tensors_list

    def numpy(
        self, aslist=False, *args, **kwargs
    ) -> Union[np.ndarray, List[np.ndarray]]:
        tensor_list = self.numpy_aslist(aslist)
        if aslist:
            return tensor_list
        tensor_numpy = np.dstack(tensor_list)
        if tensor_numpy.shape[0] == 1:
            return tensor_numpy[0]
        return tensor_numpy.transpose(2, 0, 1)

    @property
    def num_samples(self):
        return len(self.indra_tensors)

    def can_convert_to_numpy(self):
        if None in self.shape:
            return False
        return True

    def get_first_dim(self):
        return self.first_dim

    def set_first_dim(self, first_dim):
        self.first_dim = first_dim

    @property
    def max_shape(self):
        return self.indra_tensors.max_shape

    @property
    def min_shape(self):
        return self.indra_tensors.min_shape

    def callect_final_shape(self):
        shape = (self.first_dim,)
        for i, dim_len in enumerate(self.max_shape):
            if dim_len == self.min_shape[i]:
                shape += (dim_len,)
            else:
                shape += (None,)
        return shape

    @property
    def shape(self):
        first_dim = len(self.indra_tensors)
        self.set_first_dim(first_dim)
        shape = self.callect_final_shape()
        return shape

    @property
    def shape_interval(self):
        min_shape = [self.shape[0]] + self.min_shape
        max_shape = [self.shape[0]] + self.max_shape
        return shape_interval.ShapeInterval(min_shape, max_shape)

    @property
    def ndim(self):
        return len(self.indra_tensors.max_shape) + 1

    @property
    def htype(self):
        """Htype of the tensor."""
        htype = self.deeplake_tensor.meta.htype
        if self.is_sequence:
            htype = f"sequence[{htype}]"
        if self.is_link:
            htype = f"link[{htype}]"
        return htype

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
        return self.shape[0]


class DeeplakeQueryTensorWithSliceIndices(DeepLakeQueryTensor):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
        )
        self.idxs = self.index.values

    @property
    def shape(self):
        shape = ()

        max_shapes = (self.num_samples,)
        max_shapes += tuple(self.max_shape)

        for i, idx in enumerate(self.idxs):
            if isinstance(idx.value, slice):
                start = idx.value.start or 0
                stop = idx.value.stop or max_shapes[i]
                step = idx.value.step or 1

                if start < 0:
                    start = max_shapes[i] + start

                if stop < 0:
                    stop = max_shapes[i] + stop

                dim = (stop - start) // step
                shape += (dim,)
            else:
                shape += (1,)

            if i != 0 and self.max_shape[i] != self.min_shape[i]:
                raise Exception("Data across this dimension has different shapes")

        for i in range(len(self.idxs), self.ndim):
            dim_len = self.max_shape[i - 1]
            if dim_len == self.min_shape[i - 1]:
                shape += (dim_len,)
            else:
                shape += (None,)
        return shape

    def numpy_aslist(self, aslist):
        if self.min_shape != self.max_shape and aslist == False:
            raise DynamicTensorNumpyError(self.key, self.index, "shape")

        if len(self.index.values) == 1:
            tensors = self.indra_tensors[self.idxs[0].value]
        else:
            idxs = [idx.value for idx in self.idxs]

            tensors_list = self.indra_tensors[idxs[0]]
            try:
                tensors = np.dstack(tensors_list).transpose(2, 0, 1)
            except:
                idxs_tuple = tuple(idxs[1:])
                try:
                    tensors = []
                    for tensor in tensors_list:
                        tensors.append(tensor[idxs_tuple])
                    tensors = np.dstack(tensors_list).transpose(2, 0, 1)
                    return tensors
                except:
                    raise Exception(
                        "The data in the dataset has different shape across dimensions"
                    )
            idxs_tuple = (slice(None, None, None),)
            idxs_tuple += tuple(idxs[1:])
            tensors = tensors[idxs_tuple]
        return tensors


class DeeplakeQueryTensorWithIntIndices(DeepLakeQueryTensor):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
        )
        self.idx = self.index.values

    @property
    def shape(self):
        if len(self.index.values) == 1:
            first_dim = len(self.index.values)
            self.set_first_dim(first_dim)
            shape = self.callect_final_shape()
            return shape

    def numpy_aslist(self, aslist):
        idx_value = self.idx[0].value
        tensors_list = self.indra_tensors[idx_value]

        if len(self.idx) > 1:
            tensors_list = [self.indra_tensors[idx.value] for idx in self.idx]

        if self.min_shape != self.max_shape and self.aslist == False:
            raise DynamicTensorNumpyError(self.key, self.index, "shape")

        return tensors_list


class DeeplakeQueryTensorWithListIndices(DeepLakeQueryTensor):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
        )
        self.idx = self.index.values

    @property
    def shape(self):
        idx = self.index.values[0].value
        first_dim = len(idx)
        self.set_first_dim(first_dim)
        shape = self.callect_final_shape()
        return shape

    def numpy_aslist(self, aslist):
        tensors_list = [self.indra_tensors[idx] for idx in self.idx]

        if self.min_shape != self.max_shape and self.aslist == False:
            raise DynamicTensorNumpyError(self.key, self.index, "shape")

        return tensors_list


INDEX_TO_CLASS = {
    int: DeeplakeQueryTensorWithSliceIndices,
    slice: DeeplakeQueryTensorWithSliceIndices,
    list: DeeplakeQueryTensorWithListIndices,
    tuple: DeeplakeQueryTensorWithListIndices,
}


def cast_index_to_class(index):
    idx = index.values[0].value
    index_type = type(idx)
    if index_type in INDEX_TO_CLASS:
        casting_class = INDEX_TO_CLASS[index_type]
        return casting_class
    raise Exception(f"index type {index_type} is not supported")
