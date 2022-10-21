import itertools
from deeplake.core.linked_sample import LinkedSample
from deeplake.core.sample import Sample  # type: ignore
from deeplake.util.exceptions import TensorInvalidSampleShapeError
from itertools import chain
import numpy as np


class TransformTensor:
    def __init__(self, name, dataset, items=None, slice_list=None) -> None:
        self.name = name
        self.dataset = dataset
        self.items = items if items is not None else []
        self.slice_list = slice_list or []
        self.length = None
        self._ndim = None

    def _numpy(self):
        """Returns all the items stored in the slice of the tensor as numpy arrays. Even samples stored using :meth:`deeplake.read` are converted to numpy arrays in this."""
        items = self.numpy_compressed()
        for i in range(len(items)):
            if isinstance(items[i], list):
                items[i] = [x.array if isinstance(x, Sample) else x for x in items[i]]
            elif isinstance(items[i], Sample):
                items[i] = items[i].array
        return items

    def numpy(self):
        return list(itertools.chain(*self._numpy()))

    @property
    def flat_items(self):
        return list(chain(*self.items))

    def numpy_compressed(self):
        """Returns all the items stored in the slice of the tensor. Samples stored using :meth:`deeplake.read` are not converted to numpy arrays in this."""
        items = self.items[:]
        for slice_ in self.slice_list:
            for i in range(len(items)):
                items[i] = items[i][slice_]
        self.length = sum(
            map(lambda v: len(v) if isinstance(v, (np.ndarray, list)) else 1, items)
        )
        return items

    def __len__(self) -> int:
        if not self.slice_list:
            return len(self.items)
        if self.length is None:
            self.numpy_compressed()  # calculates and sets length
        return self.length  # type: ignore

    def __getattr__(self, name):
        # Only applicable for tensor groups
        if self.items:
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

    def append(self, item):
        """Adds an item to the tensor."""
        if not isinstance(item, LinkedSample):
            shape = getattr(item, "shape", None)
            if shape is None:
                item = np.asarray(item)
                shape = item.shape
            if self._ndim is None:
                self._ndim = len(shape)
            else:
                if len(shape) != self._ndim:
                    raise TensorInvalidSampleShapeError(shape, self._ndim)

        self.items.append([item])

    def extend(self, items):
        """Adds multiple items to the tensor."""
        if not isinstance(items, (np.ndarray, list)):
            items = list(items)
        self.items.append(items)
