from hub.util.shape import Shape
from typing import List, Union
import warnings

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
from hub.util.index import Index


class Tensor:
    def __init__(
        self,
        key: str,
        storage: StorageProvider,
        tensor_meta: dict = None,
        index: Union[int, slice, Index] = None,
    ):
        """Initialize a new tensor.

        Note:
            This operation does not create a new tensor in the storage provider,
            and should normally only be performed by Hub internals.

        Args:
            key (str): The internal identifier for this tensor.
            storage (StorageProvider): The storage provider for the parent dataset.
            tensor_meta (dict): For internal use only. If a tensor with `key` doesn't exist, a new tensor is created with this meta.
            index: The Index object restricting the view of this tensor.
                Can be an int, slice, or (used internally) an Index object.

        Raises:
            TensorDoesNotExistError: If no tensor with `key` exists and a `tensor_meta` was not provided.
        """
        self.key = key
        self.storage = storage
        self.index = Index(index)

        if tensor_exists(self.key, self.storage):
            if tensor_meta is not None:
                warnings.warn(
                    "Tensor should not be constructed with tensor_meta if a tensor already exists. Ignoring incoming tensor_meta. Key: {}".format(
                        self.key
                    )
                )
        else:
            if tensor_meta is None:
                raise TensorDoesNotExistError(self.key)
            create_tensor(self.key, self.storage, tensor_meta)

    def append(self, array: np.ndarray, batched: bool):
        # TODO: split into `append`/`extend`
        add_samples_to_tensor(
            array,
            self.key,
            storage=self.storage,
            batched=batched,
        )

    @property
    def meta(self):
        return read_tensor_meta(self.key, self.storage)

    @property
    def shape(self):
        ds_meta = self.meta

        min_shape = ds_meta["min_shape"]
        max_shape = ds_meta["max_shape"]

        return Shape(min_shape, max_shape)

    def __len__(self):
        """Return the length of the primary axis."""
        return self.meta["length"]

    def __getitem__(self, item: Union[int, slice, Index]):
        return Tensor(self.key, self.storage, index=self.index[item])

    def __setitem__(self, item: Union[int, slice], value: np.ndarray):
        sliced_self = self[item]
        if sliced_self.index.item != slice(None):
            raise NotImplementedError(
                "Assignment to Tensor subsections not currently supported!"
            )
        else:
            if tensor_exists(self.key, self.storage):
                raise TensorAlreadyExistsError(self.key)

            add_samples_to_tensor(
                array=value,
                key=self.key,
                storage=self.storage,
                batched=True,
            )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def numpy(self, aslist=False) -> Union[np.ndarray, List[np.ndarray]]:
        """Compute the contents of this tensor in numpy format.

        Args:
            aslist (bool): If True, a list of np.ndarrays will be returned. Helpful for dynamic tensors.
                If False, a numpy array will be returned (unless shape is dynamic).

        Returns:
            A numpy array containing the data represented by this tensor.
        """

        return read_samples_from_tensor(
            self.key, self.storage, self.index, aslist=aslist
        )
