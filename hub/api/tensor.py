from typing import Union
import warnings

import numpy as np

from hub.core.tensor import (
    create_tensor,
    add_samples_to_tensor,
    read_samples_from_tensor,
    read_tensor_meta,
    write_tensor_meta,
    tensor_exists,
)
from hub.core.typing import StorageProvider

from hub.util.exceptions import TensorAlreadyExistsError, TensorDoesNotExistError
from hub.util.index import Index


class Tensor:
    def __init__(
        self,
        key: str,
        provider: StorageProvider,
        tensor_meta: dict = None,
        index: Union[int, slice, Index] = None,
    ):
        """Initializes a new tensor.

        Note:
            This operation does not create a new tensor in the storage provider,
            and should normally only be performed by Hub internals.

        Args:
            key (str): The internal identifier for this tensor.
            provider (StorageProvider): The storage provider for the parent dataset.
            tensor_meta (dict): For internal use only. If a tensor with `key` doesn't exist, a new tensor is created with this meta.
            index: The Index object restricting the view of this tensor.
                Can be an int, slice, or (used internally) an Index object.

        Raises:
            TensorDoesNotExistError: If no tensor with `key` exists and a `tensor_meta` was not provided.
        """
        self.key = key
        self.provider = provider
        self.index = Index(index)

        if tensor_exists(self.key, self.provider):
            if tensor_meta is not None:
                warnings.warn(
                    "Tensor should not be constructed with tensor_meta if a tensor already exists. Ignoring incoming tensor_meta. Key: {}".format(
                        self.key
                    )
                )

        else:
            if tensor_meta is None:
                raise TensorDoesNotExistError(self.key)

            create_tensor(self.key, self.provider, tensor_meta)

    def extend(self, array: np.ndarray):
        """Extends a tensor by appending elements from a batched numpy array.

        Example:
            >>> len(image)
            0
            >>> image.extend(np.zeros((100, 28, 28, 1)))
            >>> len(image)
            100

        Args:
            array (np.ndarray): The data to add to the tensor.
                The primary axis should be the number of samples to add.
        """
        add_samples_to_tensor(array, self.key, storage=self.provider, batched=True)

    def append(self, array: np.ndarray):
        """Appends a sample to the end of a tensor.

        Example:
            >>> len(image)
            0
            >>> image.append(np.zeros((28, 28, 1)))
            >>> len(image)
            1

        Args:
            array (np.ndarray): The data to add to the tensor.
        """
        add_samples_to_tensor(array, self.key, storage=self.provider, batched=False)

    @property
    def meta(self):
        return read_tensor_meta(self.key, self.provider)

    @meta.setter
    def meta(self, new_meta: dict):
        write_tensor_meta(self.key, self.provider, new_meta)

    @property
    def shape(self):
        # TODO: when dynamic arrays are supported, handle `min_shape != max_shape` (right now they're always equal)
        return self.meta["max_shape"]

    def __len__(self):
        """Returns the length of the primary axis of a tensor."""
        return self.meta["length"]

    def __getitem__(self, item: Union[int, slice, Index]):
        return Tensor(self.key, self.provider, index=self.index[item])

    def __setitem__(self, item: Union[int, slice], value: np.ndarray):
        raise NotImplementedError("Tensor update not currently supported!")

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def numpy(self):
        """Computes the contents of a tensor in numpy format.

        Returns:
            A numpy array containing the data represented by this tensor.
        """
        return read_samples_from_tensor(self.key, self.provider, self.index)
