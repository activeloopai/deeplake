import deeplake.util.shape_interval as shape_interval
from deeplake.core import tensor
from typing import List, Union
from deeplake.core.index import Index
import numpy as np
import itertools
from deeplake.core.index import replace_ellipsis_with_slices
from deeplake.util.exceptions import DynamicTensorNumpyError, InvalidKeyTypeError
from deeplake.util.pretty_print import summary_tensor


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
        return len(self.indra_tensors)

    @property
    def max_shape(self):
        return tuple(self.indra_tensors.max_shape)

    @property
    def min_shape(self):
        return tuple(self.indra_tensors.min_shape)

    @property
    def shape(self):
        shape = list(self.max_shape)
        for i, dim in enumerate(self.max_shape):
            if dim != self.min_shape[i]:
                shape[i] = None
        shape = [len(self.indra_tensors)] + shape
        return tuple(shape)

    @property
    def shape_interval(self):
        min_shape = (len(self.indra_tensors),) + self.min_shape
        max_shape = (len(self.indra_tensors),) + self.max_shape
        return shape_interval.ShapeInterval(min_shape, max_shape)

    @property
    def ndim(self):
        return len(self.shape)

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


class DeeplakeQueryTensorWithSliceIndices(DeepLakeQueryTensor):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
        )
        # TO DO: Optimize this, we shouldn't be loading indra tensor here.
        self.idxs = tuple([idx.value for idx in self.index.values])
        # self.shapes = self.indra_tensors.shape

    # def _set_tensors_list(self):
    #     if len(self.index.values) == 1:
    #         self.tensors_list = self.indra_tensors[self.idxs[0]]
    #     else:
    #         self.tensors_list = self.indra_tensors[self.idxs[0]]

    # @staticmethod
    # def _find_first_dim(tensors_list):
    #     first_dim = len(tensors_list)
    #     return first_dim

    # def get_indra_tensor_shapes(self):
    #     shapes = []
    #     for idx in self.idxs:
    #         shapes += self.indra_tensors.shape(idx)
    #     return np.array(shapes)

    # @property
    # def max_shape(self):
    #     indra_tensor_shapes = self.get_indra_tensor_shapes()
    #     _max_shape = []

    #     for axis in range(len(indra_tensor_shapes)):
    #         indra_tensors_shapes_along_axis = indra_tensor_shapes[:, axis]
    #         max_value = np.max(indra_tensors_shapes_along_axis)
    #         _max_shape.append(max_value)
    #     return _max_shape

    # @property
    # def min_shape(self):
    #     indra_tensor_shapes = self.get_indra_tensor_shapes()
    #     _min_shape = []

    #     for axis in range(len(indra_tensor_shapes)):
    #         indra_tensors_shapes_along_axis = indra_tensor_shapes[:, axis]
    #         min_value = np.min(indra_tensors_shapes_along_axis)
    #         _min_shape.append(min_value)
    #     return _min_shape

    # @property
    # def shape(self):
    #     shape = ()

    #     for i, idx in enumerate(self.idxs):
    #         if isinstance(idx, int):
    #             if i > 0:
    #                 shape += (1,)
    #         else:
    #             start = idx.start or 0
    #             stop = idx.stop or self.max_shape[i]
    #             step = idx.step or 1

    #             if start < 0:
    #                 start = self.max_shape[i] + start

    #             if stop < 0:
    #                 stop = self.max_shape[i] + stop

    #             dim = (stop - start) // step
    #             shape += (dim,)

    #         if i != 0 and self.max_shape[i] != self.min_shape[i]:
    #             shape += (None,)

    #     for i in range(len(self.idxs), self.ndim):
    #         dim_len = self.max_shape[i]
    #         if dim_len == self.min_shape[i]:
    #             shape += (dim_len,)
    #         else:
    #             shape += (None,)
    #     return shape

    def get_shape(self, lst, shape=[]):
        """
        returns the shape of nested lists similarly to numpy's shape.

        :param lst: the nested list
        :param shape: the shape up to the current recursion depth
        :return: the shape including the current depth
                (finally this will be the full depth)
        """

        if not isinstance(lst, list):
            # base case
            return shape

        # peek ahead and assure all lists in the next depth
        # have the same length
        if isinstance(lst[0], list):
            l = len(lst[0])
            if not all(len(item) == l for item in lst):
                shape += [
                    None,
                ]
            else:
                shape += [
                    len(lst),
                ]

        # recurse
        shape = self.get_shape(lst[0], shape)
        return shape

    @property
    def shape(self):
        indra_tensor_shapes = self.indra_tensors[self.idxs].shape

        if list not in map(type, self.tensors_list):
            return indra_tensor_shapes

        indra_tensor_shape = self.get_shape(indra_tensor_shapes)
        return indra_tensor_shape

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


class DeeplakeQueryTensorWithIntIndices(DeepLakeQueryTensor):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
        )
        # TO DO: optimize this, we shouldn't be loading indra tensor here.
        self.idx = self.index.values
        idx_value = self.idx[0].value
        self.tensors_list = self.indra_tensors[idx_value]

    @staticmethod
    def _compare_shapes(final_shape, shapes):
        for i, dim in enumerate(shapes):
            if final_shape[i + 1] != dim:
                final_shape[i + 1] = None
        return final_shape

    def _append_first_dim(self, arr):
        arr = [len(self.tensors_list)] + list(arr.shape)
        return tuple(arr)

    @property
    def shape(self):
        # TO DO: optimize this
        shapes = {self._append_first_dim(arr) for arr in self.tensors_list}
        shape = list(shapes)[0]
        if len(shapes) > 1:
            shape_list = [len(self.tensors_list)] + list(shapes[0])
            for shape in shapes:
                if shape != shape_list[1:]:
                    shape_list = self._compare_shapes(shape_list, shape)
            shape = shape_list
        return tuple(shape)

    def numpy(self, aslist=False):
        if len(self.idx) > 1:
            self.tensors_list = [self.indra_tensors[idx.value] for idx in self.idx]

        if aslist == False:
            try:
                self.tensors_list = np.stack(self.tensors_list, axis=0)
            except:
                raise DynamicTensorNumpyError(self.key, self.index, "shape")

        return self.tensors_list


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
