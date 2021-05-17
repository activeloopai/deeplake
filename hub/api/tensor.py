from hub.util.slice import merge_slices
import numpy as np
from typing import Union
from hub.core.storage.provider import StorageProvider
from hub.core.storage.memory import MemoryProvider
from hub.core.storage.local import LocalProvider
from hub.core.chunk_engine.read import read_array, read_tensor_meta
from hub.core.chunk_engine.write import write_array, write_tensor_meta


class Tensor:
    def __init__(
        self,
        key: str,
        provider: StorageProvider,
        tensor_slice: slice = slice(None),
    ):
        """Initialize a new tensor.

        Note:
            This operation does not create a new tensor in the backend,
            and should normally only be performed by Hub internals.

        Args:
            key (str): The internal identifier for this tensor.
            provider (StorageProvider): The storage provider for the parent dataset.
            tensor_slice (slice): The slice object restricting the view of this tensor.
        """
        self.key = key
        self.provider = provider
        self.slice = tensor_slice

        self._update_from_meta()

    def _update_from_meta(self):
        meta = read_tensor_meta(self.key, self.provider)
        self.num_samples = meta["length"]
        self.shape = meta["max_shape"]

    def __len__(self):
        """Return the length of the primary axis"""
        return self.num_samples

    def __getitem__(self, item: Union[int, slice]):
        if isinstance(item, int):
            item = slice(item, item + 1)

        if isinstance(item, slice):
            new_slice = merge_slices(self.slice, item)
            return Tensor(self.key, self.provider, new_slice)

    def __setitem__(self, item: Union[int, slice], value: np.ndarray):
        sliced_self = self[item]
        if sliced_self.slice != slice(None):
            raise NotImplementedError(
                "Assignment to Tensor slices not currently supported!"
            )
        else:
            write_array(
                value,
                item,
                storage=self.provider,
                batched=True,
            )
            self._update_from_meta()

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def numpy(self):
        """Compute the contents of this tensor in numpy format.

        Returns:
            A numpy array containing the data represented by this tensor.
        """
        return read_array(self.key, self.provider, self.slice)
