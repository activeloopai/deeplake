from hub.util.slice import merge_slices
import numpy as np
from typing import Union


class Tensor:
    def __init__(self, uuid: str, tensor_slice: slice = slice(None)):
        """Initialize a new tensor.

        Note:
            This operation does not create a new tensor in the backend,
            and should normally only be performed by Hub internals.

        Args:
            uuid (str): The internal identifier for this tensor.
            tensor_slice (slice, optional): The slice object restricting the view of this tensor.
        """
        self.uuid = uuid
        self.slice = tensor_slice
        self.shape = (0,)  # Dataset should pass down relevant metadata

    def __len__(self):
        """Return the length of the primary axis"""
        return self.shape[0]

    def __getitem__(self, item: Union[int, slice]):
        if isinstance(item, int):
            item = slice(item, item + 1)

        if isinstance(item, slice):
            new_slice = merge_slices(self.slice, item)
            return Tensor(self.uuid, new_slice)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def numpy(self):
        """Compute the contents of this tensor in numpy format.

        Returns:
            A numpy array containing the data represented by this tensor.
        """
        return None  # TODO: fetch data from core
