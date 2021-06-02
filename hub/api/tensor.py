from typing import Union

import numpy as np

from hub.core.tensor import (
    create_tensor,
    add_samples_to_tensor,
    read_samples_from_tensor,
    read_tensor_meta,
    tensor_exists,
)
from hub.core.typing import StorageProvider

from hub.util.exceptions import TensorAlreadyExistsError, TensorDoesNotExistError
from hub.util.slice import merge_slices


class Tensor:
    def __init__(
        self,
        key: str,
        provider: StorageProvider,
        tensor_slice: slice = slice(None),
        tensor_meta: dict = None,
    ):
        """Initialize a new tensor.

        Note:
            This operation does not create a new tensor in the storage provider,
            and should normally only be performed by Hub internals.

        Args:
            key (str): The internal identifier for this tensor.
            provider (StorageProvider): The storage provider for the parent dataset.
            tensor_slice (slice): The slice object restricting the view of this tensor.
            tensor_meta (dict): For internal use only. If a tensor with `key` doesn't exist, a new tensor is created with this meta.

        Raises:
            TensorDoesNotExistError: If no tensor with `key` exists and a `tensor_meta` was not provided.
        """
        self.key = key
        self.provider = provider
        self.slice = tensor_slice

        if not tensor_exists(self.key, self.provider):
            if tensor_meta is None:
                raise TensorDoesNotExistError(self.key)

            create_tensor(self.key, self.provider, tensor_meta)

    def append(self, array: np.ndarray, batched: bool):
        add_samples_to_tensor(
            array,
            self.key,
            storage=self.provider,
            batched=batched,
        )

    @property
    def meta(self):
        return read_tensor_meta(self.key, self.provider)

    @property
    def shape(self):
        return self.meta["shape"]

    def __len__(self):
        """Return the length of the primary axis."""
        return self.meta["length"]

    def __getitem__(self, item: Union[int, slice]):
        if isinstance(item, int):
            item = slice(item, item + 1)

        if isinstance(item, slice):
            new_slice = merge_slices(self.slice, item)
            return Tensor(self.key, self.provider, tensor_slice=new_slice)

    def __setitem__(self, item: Union[int, slice], value: np.ndarray):
        sliced_self = self[item]
        if sliced_self.slice != slice(None):
            raise NotImplementedError(
                "Assignment to Tensor slices not currently supported!"
            )
        else:
            if tensor_exists(self.key, self.provider):
                raise TensorAlreadyExistsError(self.key)

            add_samples_to_tensor(
                array=value,
                key=self.key,
                storage=self.provider,
                batched=True,
            )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def numpy(self):
        """Compute the contents of this tensor in numpy format.

        Returns:
            A numpy array containing the data represented by this tensor.
        """
        return read_samples_from_tensor(self.key, self.provider, self.slice)
