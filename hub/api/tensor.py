from typing import Union

import numpy as np

from hub.core.chunk_engine.read import read_array, read_tensor_meta
from hub.core.chunk_engine.write import write_array
from hub.core.typing import StorageProvider
from hub.util.index import Index


class Tensor:
    def __init__(
        self,
        key: str,
        provider: StorageProvider,
        index: Union[int, slice, Index] = Index(),
    ):
        """Initialize a new tensor.

        Note:
            This operation does not create a new tensor in the storage provider,
            and should normally only be performed by Hub internals.

        Args:
            key (str): The internal identifier for this tensor.
            provider (StorageProvider): The storage provider for the parent dataset.
            index: The Index object restricting the view of this tensor.
                Can be an int, slice, or (used internally) an Index object.
        """
        self.key = key
        self.provider = provider
        self.index = Index(index)

        self.load_meta()

    def load_meta(self):
        meta = read_tensor_meta(self.key, self.provider)
        self.num_samples = meta["length"]
        self.shape = meta["max_shape"]

    def __len__(self):
        """Return the length of the primary axis"""
        return self.num_samples

    def __getitem__(self, item: Union[int, slice, Index]):
        return Tensor(self.key, self.provider, self.index[item])

    def __setitem__(self, item: Union[int, slice], value: np.ndarray):
        sliced_self = self[item]
        if sliced_self.index.item != slice(None):
            raise NotImplementedError(
                "Assignment to Tensor subsections not currently supported!"
            )
        else:
            write_array(
                array=value,
                key=self.key,
                storage=self.provider,
                batched=True,
            )
            self.load_meta()

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def numpy(self):
        """Compute the contents of this tensor in numpy format.

        Returns:
            A numpy array containing the data represented by this tensor.
        """
        return read_array(self.key, self.provider, self.index)
