import os
import warnings
from typing import Dict, Optional, Union

import numpy as np
from hub.api.tensor import Tensor

from hub.core.tensor import tensor_exists
from hub.core.dataset import dataset_exists
from hub.core.meta.dataset_meta import read_dataset_meta, write_dataset_meta
from hub.core.meta.tensor_meta import tensor_meta_from_array

from hub.core.typing import StorageProvider
from hub.util.exceptions import (
    InvalidKeyTypeError,
    TensorAlreadyExistsError,
    TensorNotFoundError,
    UnsupportedTensorTypeError,
)
from hub.util.path import provider_from_path
from hub.util.slice import merge_slices


class Dataset:
    def __init__(
        self,
        path: str = "",
        mode: str = "a",
        ds_slice: slice = slice(None),
        provider: Optional[StorageProvider] = None,
    ):
        """Initialize a new or existing dataset.

        Args:
            path (str): The location of the dataset. Used to initialize the storage provider.
            mode (str): Mode in which the dataset is opened.
                Supported modes include ("r", "w", "a") plus an optional "+" suffix.
                Defaults to "a".
            ds_slice (slice): The slice object restricting the view
                of this dataset's tensors. Defaults to slice(None, None, None).
                Used internally for iteration.
            provider (StorageProvider, optional): The storage provider used to access
                the data stored by this dataset.

        Raises:
            ValueError: If an existing local path is given, it must be a directory.
            UserWarning: Both a path and provider should not be given.
        """
        self.mode = mode
        self.slice = ds_slice

        if provider is not None and path:
            warnings.warn(
                "Dataset should not be constructed with both provider and path."
            )
        self.provider = provider or provider_from_path(path)

        self.tensors: Dict[str, Tensor] = {}

        if dataset_exists(self.provider):
            ds_meta = read_dataset_meta(self.provider)
            for tensor_name in ds_meta["tensors"]:
                self.tensors[tensor_name] = Tensor(tensor_name, self.provider)
        else:
            write_dataset_meta(self.provider, {"tensors": []})

    def __len__(self):
        """Return the greatest length of tensors"""
        return max(map(len, self.tensors.values()), default=0)

    def __getitem__(self, item: Union[slice, str, int]):
        if isinstance(item, int):
            item = slice(item, item + 1)

        if isinstance(item, str):
            if item not in self.tensors:
                raise TensorNotFoundError(item)
            else:
                return self.tensors[item][self.slice]
        elif isinstance(item, slice):
            new_slice = merge_slices(self.slice, item)
            return Dataset(mode=self.mode, ds_slice=new_slice, provider=self.provider)
        else:
            raise InvalidKeyTypeError(item)

    def __setitem__(self, item: Union[slice, str], value):
        if isinstance(item, str):
            tensor_key = item

            if tensor_exists(tensor_key, self.provider):
                raise TensorAlreadyExistsError(tensor_key)

            if isinstance(value, np.ndarray):
                batched = True

                tensor_meta = tensor_meta_from_array(value, batched)

                ds_meta = read_dataset_meta(self.provider)
                ds_meta["tensors"].append(tensor_key)
                write_dataset_meta(self.provider, ds_meta)

                tensor = Tensor(tensor_key, self.provider, tensor_meta=tensor_meta)
                self.tensors[tensor_key] = tensor
                tensor.append(value, batched=batched)

                return tensor
            else:
                raise UnsupportedTensorTypeError(item)
        else:
            raise InvalidKeyTypeError(item)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @staticmethod
    def from_path(path: str):
        """Create a local hub dataset from unstructured data.

        Note:
            This copies the data locally in hub format.
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
