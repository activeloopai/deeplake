from hub.core.sample import Sample  # type: ignore
from hub.util.exceptions import TensorInvalidSampleShapeError
import numpy as np


class TransformTensor:
    def __init__(self, base_tensor=None, slice_list=None) -> None:
        self.items = [] if base_tensor is None else base_tensor.items
        self.base_tensor = base_tensor or self
        self.slice_list = slice_list or []
        self.length = None

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
        if self.length is None:
            self.numpy_compressed()  # calculates and sets length
        return self.length  # type: ignore

    def __getitem__(self, index):
        if isinstance(index, list):
            new_slice_list = self.slice_list + index
        else:
            new_slice_list = self.slice_list + [index]
        return TransformTensor(base_tensor=self.base_tensor, slice_list=new_slice_list)

    def append(self, item):
        """Adds an item to the tensor."""
        if not isinstance(item, Sample):
            item = np.asarray(item)
            if self.items:
                expected_dims = self.items[-1].ndim
                dims = item.ndim
                if expected_dims != dims:
                    raise TensorInvalidSampleShapeError(item.shape, expected_dims)

        self.items.append(item)

    def extend(self, items):
        """Adds multiple items to the tensor."""
        for item in items:
            self.append(item)
