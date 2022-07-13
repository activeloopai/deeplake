from hub.core.linked_sample import LinkedSample
from hub.core.sample import Sample  # type: ignore
from hub.util.exceptions import TensorInvalidSampleShapeError
import numpy as np


class TransformTensor:
    def __init__(self, name, dataset, base_tensor=None, slice_list=None) -> None:
        self.name = name
        self.dataset = dataset
        self.items = [] if base_tensor is None else base_tensor.items
        self.base_tensor = base_tensor or self
        self.slice_list = slice_list or []
        self.length = None
        self._ndim = None

    def numpy(self) -> None:
        """Returns all the items stored in the slice of the tensor as numpy arrays. Even samples stored using hub.read are converted to numpy arrays in this."""
        value = self.numpy_compressed()
        if isinstance(value, list):
            for i in range(len(value)):
                if isinstance(value[i], Sample):
                    value[i] = value[i].array
        elif isinstance(value, Sample):
            value = value.array
        return value

    def numpy_compressed(self):
        """Returns all the items stored in the slice of the tensor. Samples stored using hub.read are not converted to numpy arrays in this."""
        value = self.items
        for slice_ in self.slice_list:
            value = value[slice_]
        self.length = len(value) if isinstance(value, list) else 1
        return value

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
            base_tensor=self.base_tensor,
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

        self.items.append(item)

    def extend(self, items):
        """Adds multiple items to the tensor."""
        for item in items:
            self.append(item)
