import deeplake.util.shape_interval as shape_interval
from deeplake.core import tensor
from typing import List, Union
from deeplake.core.index import Index
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
        index=None,
        is_iteration: bool = False,
        key: str = None,
    ):
        self.deeplake_tensor = deeplake_tensor
        self.indra_tensor = indra_tensor
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

        indra_tensor = self.indra_tensor
        if self.index:
            indra_tensor = indra_tensor[self.index[item]]

        return DeepLakeQueryTensor(
            self.deeplake_tensor,
            self.indra_tensor,
            index=self.index[item],
            is_iteration=is_iteration,
            key=key,
        )

    def numpy_aslist(self, aslist):
        tensors_list = self.indra_tensor[:]
        idx = self.index.values[0].value
        if idx:
            tensors_list = self.indra_tensor[idx]

        if self.min_shape != self.max_shape and aslist == False:
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
        return len(self.indra_tensor)

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
        return self.indra_tensor.max_shape

    @property
    def min_shape(self):
        return self.indra_tensor.min_shape

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
        if self.max_shape != self.min_shape:
            shape = []
            for i, dim_len in enumerate(self.max_shape):
                if dim_len == self.min_shape[i]:
                    shape.append(self.min_shape[i])
                else:
                    shape.append(None)
        return shape

    @property
    def shape_interval(self):
        return shape_interval.ShapeInterval(self.min_shape, self.max_shape)

    @property
    def ndim(self):
        return len(self.indra_tensor.max_shape)

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

    def summary(self):
        """Prints a summary of the tensor."""
        pretty_print = summary_tensor(self)

        print(self)
        print(pretty_print)


class DeeplakeQueryTensorWithSliceIndices(DeepLakeQueryTensor):  # int and slices, lists
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
        )
        # TO DO: Optimize this, we shouldn't be loading indra tensor here.
        self.idxs = [idx.value for idx in self.index.values]

    def _set_tensors_list(self):
        if len(self.index.values) == 1:
            self.tensors_list = self.indra_tensor[self.idxs[0]]
        else:
            self.tensors_list = self.indra_tensor[self.idxs[0]]

    @staticmethod
    def _find_first_dim(tensors_list):
        first_dim = len(tensors_list)
        return first_dim

    def get_indra_tensor_shapes(self):
        shapes = []
        for idx in self.idxs:
            shapes += self.indra_tensor.shape(idx)
        return np.array(shapes)

    @property
    def max_shape(self):
        indra_tensor_shapes = self.get_indra_tensor_shapes()
        _max_shape = []

        for axis in range(len(indra_tensor_shapes)):
            indra_tensors_shapes_along_axis = indra_tensor_shapes[:, axis]
            max_value = np.max(indra_tensors_shapes_along_axis)
            _max_shape.append(max_value)
        return _max_shape

    @property
    def min_shape(self):
        indra_tensor_shapes = self.get_indra_tensor_shapes()
        _min_shape = []

        for axis in range(len(indra_tensor_shapes)):
            indra_tensors_shapes_along_axis = indra_tensor_shapes[:, axis]
            min_value = np.min(indra_tensors_shapes_along_axis)
            _min_shape.append(min_value)
        return _min_shape

    @property
    def shape(self):
        shape = ()

        for i, idx in enumerate(self.idxs):
            if isinstance(idx, int):
                if i > 0:
                    shape += (1,)
            else:
                start = idx.start or 0
                stop = idx.stop or self.max_shape[i]
                step = idx.step or 1

                if start < 0:
                    start = self.max_shape[i] + start

                if stop < 0:
                    stop = self.max_shape[i] + stop

                dim = (stop - start) // step
                shape += (dim,)

            if i != 0 and self.max_shape[i] != self.min_shape[i]:
                shape += (None,)

        for i in range(len(self.idxs), self.ndim):
            dim_len = self.max_shape[i]
            if dim_len == self.min_shape[i]:
                shape += (dim_len,)
            else:
                shape += (None,)
        return shape

    def numpy(self, aslist=False):
        self._set_tensors_list()
        # TO DO: optimize this
        if self.min_shape != self.max_shape and aslist == False or None in self.shape:
            raise DynamicTensorNumpyError(self.key, self.index, "shape")

        if len(self.index.values) == 1:
            return np.stack(self.tensors_list, axis=0)

        if list not in map(type, self.tensors_list):
            tensors = np.stack(self.tensors_list, axis=0)
            idxs_tuple = tuple(self.idxs[1:])
            tensors = tensors[idxs_tuple]
            return tensors

        tensors = []
        for tensor in self.tensors_list:
            tensor = tensor[self.idxs[1]]
            tensor = np.stack(tensor, axis=-1)
            if self.idxs[2:]:
                tensors.append(np.moveaxis(tensor[tuple(self.idxs[2:])], -1, 0))
            else:
                tensors.append(np.moveaxis(tensor, -1, 0))

        tensors = np.stack(tensors, axis=0)
        return tensors
