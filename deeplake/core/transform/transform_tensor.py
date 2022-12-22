from itertools import chain
from deeplake.core.linked_sample import LinkedSample
from deeplake.core.sample import Sample  # type: ignore
from deeplake.core.tensor import Tensor
from deeplake.util.exceptions import TensorInvalidSampleShapeError
from deeplake.util.array_list import slice_array_list
import numpy as np


class TransformTensor:
    def __init__(self, name, dataset, items=None, slice_list=None) -> None:
        self.name = name
        self.dataset = dataset
        self.items = items if items is not None else []
        self.slice_list = slice_list or []
        self.length = None
        self._ndim = None
        if not self.items:
            self.items.append(True)

    @property
    def _numpy_only(self):
        return self.items[0]

    @_numpy_only.setter
    def _numpy_only(self, val):
        self.items[0] = val

    def _indexed_numpy_array_list(self):
        sub_sample_indexed = False
        val = self.items[1:]
        arr = False
        for s in self.slice_list:
            if isinstance(val, np.ndarray):
                val = val[s]
                if arr:
                    sub_sample_indexed = True
                arr = True
            else:
                val = slice_array_list(val, s)
        if sub_sample_indexed:
            self.length = 1
            return [val]
        else:
            self.length = sum(map(lambda x: getattr(x, "__len__", lambda: 1)(), val))
            return val

    def numpy(self) -> None:
        """Returns all the items stored in the slice of the tensor as numpy arrays. Even samples stored using :meth:`deeplake.read` are converted to numpy arrays in this."""
        if self._numpy_only:
            return self._indexed_numpy_array_list()
        value = self.numpy_compressed()
        if isinstance(value, list):
            for i in range(len(value)):
                if isinstance(value[i], Sample):
                    value[i] = value[i].array
        elif isinstance(value, Sample):
            value = value.array
        return value

    def numpy_compressed(self):
        """Returns all the items stored in the slice of the tensor. Samples stored using :meth:`deeplake.read` are not converted to numpy arrays in this."""
        if self._numpy_only:
            return self._indexed_numpy_array_list()
        value = self.items[1:]
        for slice_ in self.slice_list:
            value = value[slice_]
        self.length = len(value) if isinstance(value, list) else 1
        return value

    def __len__(self) -> int:
        if not self.slice_list:
            if self._numpy_only:
                return sum(
                    map(lambda x: getattr(x, "__len__", lambda: 1)(), self.items[1:])
                )
            return len(self.items[1:])
        if self.length is None:
            self.numpy_compressed()  # calculates and sets length
        return self.length  # type: ignore

    def __getattr__(self, name):
        # Only applicable for tensor groups
        if len(self.items) > 1:
            # Samples appended to this tensor, which means this is not a tensor group
            raise AttributeError(name)
        del self.dataset.tensors[self.name]
        return self.dataset["/".join((self.name, name))]

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.__getattr__(index)

        if isinstance(index, list):
            new_slice_list = self.slice_list + index
        else:
            new_slice_list = self.slice_list + [index]
        return TransformTensor(
            name=self.name,
            dataset=self.dataset,
            items=self.items,
            slice_list=new_slice_list,
        )

    def _non_numpy(self):
        new_items = list(chain(*self.items[1:]))
        self.items.clear()
        self.items.append(False)
        self.items += new_items

    def append(self, item):
        """Adds an item to the tensor."""
        if self._numpy_only:
            if isinstance(item, np.ndarray):
                return self.extend(np.expand_dims(item, 0))
            else:
                self._non_numpy()
        if not isinstance(item, (LinkedSample, Tensor)):
            shape = getattr(item, "shape", None)
            if shape is None:
                try:
                    item = np.asarray(item)
                    shape = item.shape
                except ValueError:
                    shape = (1,)
            if self._ndim is None:
                self._ndim = len(shape)
            else:
                if len(shape) != self._ndim:
                    raise TensorInvalidSampleShapeError(shape, self._ndim)

        self.items.append(item)

    def extend(self, items):
        """Adds multiple items to the tensor."""
        if self._numpy_only:
            if isinstance(items, np.ndarray) or (
                isinstance(items, list)
                and set(map(type, items)) in [{dict}, {np.ndarray}]
            ):
                try:
                    incoming_ndim = items.ndim
                    incoming_shape = items.shape
                except AttributeError:
                    try:
                        item = items[0]
                        incoming_ndim = item.ndim
                        incoming_shape = item.shape
                    except IndexError:
                        incoming_ndim = None
                        incoming_shape = None
                if incoming_ndim:
                    if self._ndim is None:
                        self._ndim = incoming_ndim
                    else:
                        if self._ndim != incoming_ndim:
                            raise TensorInvalidSampleShapeError(
                                incoming_shape, self._ndim
                            )
                return self.items.append(items)
            else:
                self._non_numpy()

        for item in items:
            self.append(item)
