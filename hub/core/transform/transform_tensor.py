from hub.core.chunk_engine import is_uniform_sequence
from hub.util.exceptions import TensorInvalidSampleShapeError
import numpy as np


class TransformDatasetTensor:
    def __init__(self, base_tensor=None, slice_list=None) -> None:
        self.items = [] if base_tensor is None else base_tensor.items
        self.base_tensor = base_tensor or self
        self.slice_list = slice_list or []

    def numpy(self) -> None:
        value = self.items
        for slice_ in self.slice_list:
            value = value[slice_]
        if isinstance(value, list) and is_uniform_sequence(value):
            return np.array(value)
        return value

    def __len__(self) -> int:
        return len(self.numpy())

    def __getitem__(self, index):
        if isinstance(index, list):
            new_slice_list = self.slice_list + index
        else:
            new_slice_list = self.slice_list + [index]
        return TransformDatasetTensor(
            base_tensor=self.base_tensor, slice_list=new_slice_list
        )

    def append(self, item):
        item = np.array(item)
        if self.items:
            expected_dims = self.items[-1].ndim
            dims = item.ndim
            if expected_dims != dims:
                raise TensorInvalidSampleShapeError(
                    f"Sample shape length is expected to be {expected_dims}, actual length is {dims}.",
                    item.shape,
                )
        self.items.append(item)

    def extend(self, items):
        for item in items:
            self.append(item)
