"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from hub.utils import _tuple_product
from hub.api.tensorview import TensorView
import collections.abc as abc
from hub.api.dataset_utils import (
    create_numpy_dict,
    get_value,
    slice_split,
    str_to_int,
)
from hub.exceptions import LargeShapeFilteringException, NoneValueException
from hub.api.objectview import ObjectView
from hub.schema import Sequence


class DatasetView:
    def __init__(
        self,
        dataset=None,
        lazy: bool = True,
        indexes=None,  # list or integer
    ):
        """Creates a DatasetView object for a subset of the Dataset.

        Parameters
        ----------
        dataset: hub.api.dataset.Dataset object
            The dataset whose DatasetView is being created
        lazy: bool, optional
            Setting this to False will stop lazy computation and will allow items to be accessed without .compute()
        indexes: optional
            It can be either a list or an integer depending upon the slicing. Represents the indexes that the datasetview is representing.
        """
        if dataset is None:
            raise NoneValueException("dataset")
        if indexes is None:
            raise NoneValueException("indexes")

        self.dataset = dataset
        self.lazy = lazy
        self.indexes = indexes
        self.is_contiguous = False
        if isinstance(self.indexes, list) and self.indexes:
            self.is_contiguous = self.indexes[-1] - self.indexes[0] + 1 == len(
                self.indexes
            )

    def __getitem__(self, slice_):
        """| Gets a slice or slices from DatasetView
        | Usage:

        >>> ds_view = ds[5:15]
        >>> return ds_view["image", 7, 0:1920, 0:1080, 0:3].compute() # returns numpy array of 12th image
        """
        if not isinstance(slice_, abc.Iterable) or isinstance(slice_, str):
            slice_ = [slice_]
        slice_ = list(slice_)
        subpath, slice_list = slice_split(slice_)
        slice_list = [0] + slice_list if isinstance(self.indexes, int) else slice_list
        if not subpath:
            if len(slice_list) > 1:
                raise ValueError("Can't slice dataset with multiple slices without key")
            indexes = self.indexes[slice_list[0]]
            return DatasetView(dataset=self.dataset, lazy=self.lazy, indexes=indexes)
        elif not slice_list:
            slice_ = (
                [slice(self.indexes[0], self.indexes[-1] + 1)]
                if self.is_contiguous
                else [self.indexes]
            )
            if subpath in self.keys:
                tensorview = TensorView(
                    dataset=self.dataset,
                    subpath=subpath,
                    slice_=slice_,
                    lazy=self.lazy,
                )
                return tensorview if self.lazy else tensorview.compute()
            for key in self.keys:
                if subpath.startswith(key):
                    objectview = ObjectView(
                        dataset=self.dataset,
                        subpath=subpath,
                        slice_=slice_,
                        lazy=self.lazy,
                    )
                    return objectview if self.lazy else objectview.compute()
            return self._get_dictionary(subpath, slice_)
        else:
            if isinstance(self.indexes, list):
                indexes = self.indexes[slice_list[0]]
                if self.is_contiguous and isinstance(indexes, list) and indexes:
                    indexes = slice(indexes[0], indexes[-1] + 1)
            else:
                indexes = self.indexes
            slice_list[0] = indexes
            schema_obj = self.dataset.schema.dict_[subpath.split("/")[1]]

            if subpath in self.keys and (
                not isinstance(schema_obj, Sequence) or len(slice_list) <= 1
            ):
                tensorview = TensorView(
                    dataset=self.dataset,
                    subpath=subpath,
                    slice_=slice_list,
                    lazy=self.lazy,
                )
                return tensorview if self.lazy else tensorview.compute()
            for key in self.keys:
                if subpath.startswith(key):
                    objectview = ObjectView(
                        dataset=self.dataset,
                        subpath=subpath,
                        slice_=slice_list,
                        lazy=self.lazy,
                    )
                    return objectview if self.lazy else objectview.compute()
            if len(slice_list) > 1:
                raise ValueError("You can't slice a dictionary of Tensors")
            return self._get_dictionary(subpath, slice_list[0])

    def __setitem__(self, slice_, value):
        """| Sets a slice or slices with a value
        | Usage:

        >>> ds_view = ds[5:15]
        >>> ds_view["image", 3, 0:1920, 0:1080, 0:3] = np.zeros((1920, 1080, 3), "uint8") # sets the 8th image
        """
        self.dataset._auto_checkout()
        assign_value = get_value(value)
        assign_value = str_to_int(
            assign_value, self.dataset.tokenizer
        )  # handling strings and bytes

        if not isinstance(slice_, abc.Iterable) or isinstance(slice_, str):
            slice_ = [slice_]
        slice_ = list(slice_)
        subpath, slice_list = slice_split(slice_)
        slice_list = [0] + slice_list if isinstance(self.indexes, int) else slice_list

        if not subpath:
            raise ValueError("Can't assign to dataset sliced without key")
        elif subpath not in self.keys:
            raise KeyError(f"Key {subpath} not found in dataset")

        if not slice_list:
            slice_ = (
                slice(self.indexes[0], self.indexes[-1] + 1)
                if self.is_contiguous
                else self.indexes
            )
            if not isinstance(slice_, list):
                self.dataset._tensors[subpath][slice_] = assign_value
            else:
                for i, index in enumerate(slice_):
                    self.dataset._tensors[subpath][index] = assign_value[i]
        else:
            if isinstance(self.indexes, list):
                indexes = self.indexes[slice_list[0]]
                if self.is_contiguous and isinstance(indexes, list) and indexes:
                    slice_list[0] = slice(indexes[0], indexes[-1] + 1)
                else:
                    slice_list[0] = indexes
            else:
                slice_list[0] = self.indexes

            if not isinstance(slice_list[0], list):
                self.dataset._tensors[subpath][slice_list] = assign_value
            else:
                for i, index in enumerate(slice_list[0]):
                    current_slice = [index] + slice_list[1:]
                    self.dataset._tensors[subpath][current_slice] = assign_value[i]

    def filter(self, dic):
        """| Applies a filter to get a new datasetview that matches the dictionary provided

        Parameters
        ----------
        dic: dictionary
            A dictionary of key value pairs, used to filter the dataset. For nested schemas use flattened dictionary representation
            i.e instead of {"abc": {"xyz" : 5}} use {"abc/xyz" : 5}
        """
        indexes = self.indexes
        for k, v in dic.items():
            k = k if k.startswith("/") else "/" + k
            if k not in self.keys:
                raise KeyError(f"Key {k} not found in the dataset")
            tsv = self.dataset[k]
            max_shape = tsv.dtype.max_shape
            prod = _tuple_product(max_shape)
            if prod > 100:
                raise LargeShapeFilteringException(k)
            if isinstance(indexes, list):
                indexes = [index for index in indexes if tsv[index].compute() == v]
            else:
                indexes = indexes if tsv[indexes].compute() == v else []
        return DatasetView(dataset=self.dataset, lazy=self.lazy, indexes=indexes)

    @property
    def keys(self):
        """
        Get Keys of the dataset
        """
        return self.dataset._tensors.keys()

    def _get_dictionary(self, subpath, slice_):
        """Gets dictionary from dataset given incomplete subpath"""
        tensor_dict = {}
        subpath = subpath if subpath.endswith("/") else subpath + "/"
        for key in self.keys:
            if key.startswith(subpath):
                suffix_key = key[len(subpath) :]
                split_key = suffix_key.split("/")
                cur = tensor_dict
                for sub_key in split_key[:-1]:
                    if sub_key not in cur.keys():
                        cur[sub_key] = {}
                    cur = cur[sub_key]
                tensorview = TensorView(
                    dataset=self.dataset,
                    subpath=key,
                    slice_=slice_,
                    lazy=self.lazy,
                )
                cur[split_key[-1]] = tensorview if self.lazy else tensorview.compute()
        if not tensor_dict:
            raise KeyError(f"Key {subpath} was not found in dataset")
        return tensor_dict

    def __iter__(self):
        """ Returns Iterable over samples """
        if isinstance(self.indexes, int):
            yield self
            return

        for i in range(len(self.indexes)):
            yield self[i]

    def __len__(self):
        return len(self.indexes) if isinstance(self.indexes, list) else 1

    def __str__(self):
        return "DatasetView(" + str(self.dataset) + ")"

    def __repr__(self):
        return self.__str__()

    def to_tensorflow(self):
        """Converts the dataset into a tensorflow compatible format"""
        return self.dataset.to_tensorflow(indexes=self.indexes)

    def to_pytorch(
        self,
        transform=None,
        inplace=True,
        output_type=dict,
    ):
        """Converts the dataset into a pytorch compatible format"""
        return self.dataset.to_pytorch(
            transform=transform,
            indexes=self.indexes,
            inplace=inplace,
            output_type=output_type,
        )

    def resize_shape(self, size: int) -> None:
        """Resize dataset shape, not DatasetView"""
        self.dataset.resize_shape(size)

    def commit(self, message="") -> None:
        """Commit dataset"""
        self.dataset.commit(message)

    def flush(self) -> None:
        """Flush dataset"""
        self.dataset.flush()

    def numpy(self, label_name=False):
        """Gets the value from different tensorview objects in the datasetview schema

        Parameters
        ----------
        label_name: bool, optional
            If the TensorView object is of the ClassLabel type, setting this to True would retrieve the label names
            instead of the label encoded integers, otherwise this parameter is ignored.
        """
        if isinstance(self.indexes, int):
            return create_numpy_dict(self.dataset, self.indexes, label_name=label_name)
        else:
            return [
                create_numpy_dict(self.dataset, index, label_name=label_name)
                for index in self.indexes
            ]

    def disable_lazy(self):
        self.lazy = False

    def enable_lazy(self):
        self.lazy = True

    def compute(self, label_name=False):
        """Gets the value from different tensorview objects in the datasetview schema

        Parameters
        ----------
        label_name: bool, optional
            If the TensorView object is of the ClassLabel type, setting this to True would retrieve the label names
            instead of the label encoded integers, otherwise this parameter is ignored.
        """
        return self.numpy(label_name=label_name)
