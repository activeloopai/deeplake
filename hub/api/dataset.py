import os
import warnings
from typing import Dict, Optional, Union

import numpy as np
from hub.api.tensor import Tensor

from hub.core.tensor import tensor_exists
from hub.core.dataset import dataset_exists
from hub.core.meta.dataset_meta import read_dataset_meta, write_dataset_meta
from hub.core.meta.tensor_meta import default_tensor_meta

from hub.core.typing import StorageProvider
from hub.util.index import Index

from hub.constants import DEFAULT_CHUNK_SIZE
from hub.util.path import provider_from_path
from hub.util.exceptions import (
    InvalidKeyTypeError,
    TensorAlreadyExistsError,
    TensorDoesNotExistError,
    UnsupportedTensorTypeError,
)
from hub.util.path import provider_from_path


class Dataset:
    def __init__(
        self,
        path: str = "",
        mode: str = "a",
        provider: Optional[StorageProvider] = None,
        index: Union[int, slice, Index] = None,
    ):
        """Initialize a new or existing dataset.

        Args:
            path (str): The location of the dataset. Used to initialize the storage provider.
            mode (str): Mode in which the dataset is opened.
                Supported modes include ("r", "w", "a") plus an optional "+" suffix.
                Defaults to "a".
            provider (StorageProvider, optional): The storage provider used to access
                the data stored by this dataset.
            index: The Index object restricting the view of this dataset's tensors.
                Can be an int, slice, or (used internally) an Index object.

        Raises:
            ValueError: If an existing local path is given, it must be a directory.
            UserWarning: Both a path and provider should not be given.
        """
        self.mode = mode
        self.index = Index(index)

        if provider is not None and path:
            warnings.warn(
                "Dataset should not be constructed with both provider and path."
            )
        self.provider = provider or provider_from_path(path)

        self.tensors: Dict[str, Tensor] = {}

        if dataset_exists(self.provider):
            for tensor_name in self.meta["tensors"]:
                self.tensors[tensor_name] = Tensor(tensor_name, self.provider)
        else:
            self.meta = {"tensors": []}

    def __len__(self):
        """Return the greatest length of tensors"""
        return max(map(len, self.tensors.values()), default=0)

    def __getitem__(self, item: Union[str, int, slice, Index]):
        if isinstance(item, str):
            if item not in self.tensors:
                raise TensorDoesNotExistError(item)
            else:
                return self.tensors[item][self.index]
        elif isinstance(item, (int, slice, Index)):
            new_index = self.index[Index(item)]
            return Dataset(mode=self.mode, provider=self.provider, index=new_index)
        else:
            raise InvalidKeyTypeError(item)

    def create_tensor(
        self, name: str, chunk_size: int = DEFAULT_CHUNK_SIZE, dtype: str = "float64"
    ):
        """Create a new tensor in this dataset.

        Args:
            name (str): The name of the tensor to be created.
            chunk_size (int): The target size for chunks in this tensor.
            dtype (str): The dtype to use for this tensor.
                Will be overwritten when the first sample is added.

        Returns:
            The new tensor, which can also be accessed by `self[name]`.

        Raises:
            TensorAlreadyExistsError: Duplicate tensors are not allowed.
        """
        if tensor_exists(name, self.provider):
            raise TensorAlreadyExistsError(name)

        ds_meta = self.meta
        ds_meta["tensors"].append(name)
        self.meta = ds_meta

        tensor_meta = default_tensor_meta(chunk_size, dtype)
        tensor = Tensor(name, self.provider, tensor_meta=tensor_meta)
        self.tensors[name] = tensor

        return tensor

    __getattr__ = __getitem__

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @property
    def meta(self):
        return read_dataset_meta(self.provider)

    @meta.setter
    def meta(self, new_meta: dict):
        write_dataset_meta(self.provider, new_meta)

    @staticmethod
    def from_path(path: str):
        """Create a hub dataset from unstructured data.

        Note:
            This copies the data into hub format.
            Be careful when using this with large datasets.

        Args:
            path (str): Path to the data to be converted

        Returns:
            A Dataset instance whose path points to the hub formatted
            copy of the data.

        Raises:
            NotImplementedError: TODO.
        """

        raise NotImplementedError(
            "Automatic dataset ingestion is not yet supported."
        )  # TODO: hub.auto
        return None
