"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
import numpy as np
from typing import Iterable
import hub_v1
import collections.abc as abc
from hub_v1.api.dataset_utils import (
    get_value,
    slice_split,
    str_to_int,
    check_class_label,
)
from hub_v1.exceptions import NoneValueException
from hub_v1.schema import ClassLabel, Text, SchemaDict
import hub_v1.api.objectview as objv


class TensorView:
    def __init__(
        self,
        dataset=None,
        subpath=None,
        slice_=None,
        lazy: bool = True,
    ):
        """Creates a TensorView object for a particular tensor in the dataset

        Parameters
        ----------
        dataset: hub.api.dataset.Dataset object
            The dataset whose TensorView is being created
        subpath: str
            The full path to the particular Tensor in the Dataset
        slice_: optional
            The `slice_` of this Tensor that needs to be accessed
        lazy: bool, optional
            Setting this to False will stop lazy computation and will allow items to be accessed without .compute()
        """

        if dataset is None:
            raise NoneValueException("dataset")
        if subpath is None:
            raise NoneValueException("subpath")

        self.dataset = dataset
        self.subpath = subpath
        self.lazy = lazy

        if isinstance(slice_, (int, slice)):
            self.slice_ = [slice_]
        elif isinstance(slice_, (tuple, list)):
            self.slice_ = list(slice_)
        self.nums = [None]
        self.offsets = [None]
        self.squeeze_dims = [False]
        self.indexes = self.slice_[0]  # int, slice or list
        self.is_contiguous = False
        if isinstance(self.indexes, list) and self.indexes:
            self.is_contiguous = self.indexes[-1] - self.indexes[0] + 1 == len(
                self.indexes
            )
        for it in self.slice_[1:]:
            if isinstance(it, int):
                self.nums.append(1)
                self.offsets.append(it)
                self.squeeze_dims.append(True)
            elif isinstance(it, slice):
                ofs = it.start or 0
                num = it.stop - ofs if it.stop else None
                self.nums.append(num)
                self.offsets.append(ofs)
                self.squeeze_dims.append(False)
        self.dtype = self.dtype_from_path(subpath)
        self.shape = self.dataset._tensors[self.subpath].get_shape(self.slice_)

    def numpy(self, label_name=False):
        """Gets the value from tensorview

        Parameters
        ----------
        label_name: bool, optional
            If the TensorView object is of the ClassLabel type, setting this to True would retrieve the label names
            instead of the label encoded integers, otherwise this parameter is ignored.
        """
        if isinstance(self.indexes, list):
            value = np.array(
                [
                    self.dataset._tensors[self.subpath][[index] + self.slice_[1:]]
                    for index in self.indexes
                ]
            )
        else:
            value = self.dataset._tensors[self.subpath][self.slice_]

        if isinstance(self.dtype, hub_v1.schema.class_label.ClassLabel) and label_name:
            if isinstance(self.indexes, int):
                if value.ndim == 0:
                    value = self.dtype.int2str(value)
                elif value.ndim == 1:
                    value = [self.dtype.int2str(value[i]) for i in range(value.size)]
            else:
                if value.ndim == 1:
                    value = [self.dtype.int2str(value[i]) for i in range(value.size)]
                elif value.ndim == 2:
                    value = [
                        [self.dtype.int2str(item[i]) for i in range(item.size)]
                        for item in value
                    ]

        if isinstance(self.dtype, hub_v1.schema.text.Text):
            if self.dataset.tokenizer is not None:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
                if value.ndim == 1:
                    return tokenizer.decode(value.tolist())
                elif value.ndim == 2:
                    return [tokenizer.decode(val.tolist()) for val in value]
            elif value.ndim == 1:
                return "".join(chr(it) for it in value.tolist())
            elif value.ndim == 2:
                return ["".join(chr(it) for it in val.tolist()) for val in value]
            raise ValueError(f"Unexpected value with shape for text {value.shape}")
        return value

    def compute(self, label_name=False):
        """Gets the value from tensorview

        Parameters
        ----------
        label_name: bool, optional
            If the TensorView object is of the ClassLabel type, setting this to True would retrieve the label names
            instead of the label encoded integers, otherwise this parameter is ignored.
        """
        return self.numpy(label_name=label_name)

    def __getitem__(self, slice_):
        """| Gets a slice or slices from tensorview
        | Usage:

        >>> images_tensorview = ds["image"]
        >>> return images_tensorview[7, 0:1920, 0:1080, 0:3].compute() # returns numpy array of 7th image
        """
        if not isinstance(slice_, abc.Iterable) or isinstance(slice_, str):
            slice_ = [slice_]
        slice_ = list(slice_)
        slice_ = self.slice_fill(slice_)
        subpath, slice_list = slice_split(slice_)
        new_nums = self.nums.copy()
        new_offsets = self.offsets.copy()
        if isinstance(self.indexes, list):
            new_indexes = self.indexes[slice_list[0]]
            if self.is_contiguous and new_indexes:
                new_indexes = slice(new_indexes[0], new_indexes[-1] + 1)
        elif isinstance(self.indexes, int):
            new_indexes = self.indexes
        else:
            ofs = self.indexes.start or 0
            num = self.indexes.stop - ofs if self.indexes.stop else None
            new_indexes = self._combine(slice_list[0], num, ofs)
        slice_list[0] = new_indexes
        # new_indexes = self.indexes[]
        if len(new_nums) < len(slice_list):
            new_nums.extend([None] * (len(slice_list) - len(new_nums)))
            new_offsets.extend([0] * (len(slice_list) - len(new_offsets)))
        for i in range(1, len(slice_list)):
            slice_list[i] = self._combine(slice_list[i], new_nums[i], new_offsets[i])
        for i in range(len(slice_list), len(new_nums)):
            cur_slice = (
                slice(new_offsets[i], new_offsets[i] + new_nums[i])
                if not self.squeeze_dims[i]
                else new_offsets[i]
            )
            slice_list.append(cur_slice)
        if subpath or (
            len(slice_list) > len(self.nums) and isinstance(self.dtype, objv.Sequence)
        ):
            objectview = objv.ObjectView(
                dataset=self.dataset,
                subpath=self.subpath + subpath,
                slice_=slice_list,
                lazy=self.lazy,
            )
            return objectview if self.lazy else objectview.compute()
        else:
            tensorview = TensorView(
                dataset=self.dataset,
                subpath=self.subpath,
                slice_=slice_list,
                lazy=self.lazy,
            )
            return tensorview if self.lazy else tensorview.compute()

    def __setitem__(self, slice_, value):
        """| Sets a slice or slices with a value
        | Usage:

        >>> images_tensorview = ds["image"]
        >>> images_tensorview[7, 0:1920, 0:1080, 0:3] = np.zeros((1920, 1080, 3), "uint8") # sets 7th image
        """
        self.dataset._auto_checkout()

        if not isinstance(slice_, abc.Iterable) or isinstance(slice_, str):
            slice_ = [slice_]
        slice_ = list(slice_)
        slice_ = self.slice_fill(slice_)
        subpath, slice_list = slice_split(slice_)
        if subpath:
            raise ValueError("Can't setitem of TensorView with subpath")

        assign_value = get_value(value)
        schema_dict = self.dataset.schema
        if subpath[1:] in schema_dict.dict_.keys():
            schema_key = schema_dict.dict_.get(subpath[1:], None)
        else:
            for schema_key in subpath[1:].split("/"):
                schema_dict = schema_dict.dict_.get(schema_key, None)
                if not isinstance(schema_dict, SchemaDict):
                    schema_key = schema_dict
        if isinstance(schema_key, ClassLabel):
            assign_value = check_class_label(assign_value, schema_key)
        if isinstance(schema_key, (Text, bytes)) or (
            isinstance(assign_value, Iterable)
            and any(isinstance(val, str) for val in assign_value)
        ):
            # handling strings and bytes
            assign_value = str_to_int(assign_value, self.dataset.tokenizer)

        new_nums = self.nums.copy()
        new_offsets = self.offsets.copy()
        if isinstance(self.indexes, list):
            new_indexes = self.indexes[slice_list[0]]
            if self.is_contiguous and new_indexes:
                new_indexes = slice(new_indexes[0], new_indexes[-1] + 1)
        elif isinstance(self.indexes, int):
            new_indexes = self.indexes
        else:
            ofs = self.indexes.start or 0
            num = self.indexes.stop - ofs if self.indexes.stop else None
            new_indexes = self._combine(slice_list[0], num, ofs)
        slice_list[0] = new_indexes
        if len(new_nums) < len(slice_list):
            new_nums.extend([None] * (len(slice_list) - len(new_nums)))
            new_offsets.extend([0] * (len(slice_list) - len(new_offsets)))
        for i in range(1, len(slice_list)):
            slice_list[i] = self._combine(slice_list[i], new_nums[i], new_offsets[i])
        for i in range(len(slice_list), len(new_nums)):
            cur_slice = (
                slice(new_offsets[i], new_offsets[i] + new_nums[i])
                if not self.squeeze_dims[i]
                else new_offsets[i]
            )
            slice_list.append(cur_slice)

        if isinstance(slice_list[0], (int, slice)):
            self.dataset._tensors[self.subpath][slice_list] = assign_value
        else:
            for i, index in enumerate(slice_list[0]):
                current_slice = [index] + slice_list[1:]
                self.dataset._tensors[subpath][current_slice] = assign_value[i]

    def _combine(self, slice_, num=None, ofs=0):
        "Combines a `slice_` with the current num and offset present in tensorview"
        if isinstance(slice_, int):
            self.check_slice_bounds(num=num, start=slice_)
            return ofs + slice_
        elif isinstance(slice_, slice):
            self.check_slice_bounds(
                num=num, start=slice_.start, stop=slice_.stop, step=slice_.step
            )
            if slice_.start is None and slice_.stop is None:
                return slice(ofs, None) if num is None else slice(ofs, ofs + num)
            elif slice_.stop is None:
                return (
                    slice(ofs + slice_.start, None)
                    if num is None
                    else slice(ofs + slice_.start, ofs + num)
                )
            elif slice_.start is None:
                return slice(ofs, ofs + slice_.stop)
            else:
                return slice(ofs + slice_.start, ofs + slice_.stop)
        else:
            raise TypeError(
                "type {} isn't supported in dataset slicing".format(type(slice_))
            )

    def check_slice_bounds(self, num=None, start=None, stop=None, step=None):
        "Checks whether the bounds of slice are in limits"
        if step and step < 0:  # negative step not supported
            raise ValueError("Negative step not supported in dataset slicing")
        if num and ((start and start >= num) or (stop and stop > num)):
            raise IndexError(
                "index out of bounds for dimension with length {}".format(num)
            )
        if start and stop and start > stop:
            raise IndexError("start index is greater than stop index")

    def dtype_from_path(self, path):
        "Gets the dtype of the Tensorview by traversing the schema"
        path = path.split("/")
        cur_type = self.dataset.schema.dict_
        for subpath in path[1:-1]:
            cur_type = cur_type[subpath]
            cur_type = cur_type.dict_
        return cur_type[path[-1]]

    def slice_fill(self, slice_):
        "Fills the slice with zeroes for the dimensions that have single elements and squeeze_dims true"
        if isinstance(self.indexes, int):
            new_slice_ = [0]
            offset = 0
        else:
            new_slice_ = [slice_[0]]
            offset = 1
        for i in range(1, len(self.nums)):
            if self.squeeze_dims[i]:
                new_slice_.append(0)
            elif offset < len(slice_):
                new_slice_.append(slice_[offset])
                offset += 1
        new_slice_ += slice_[offset:]
        return new_slice_

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return (
            "TensorView("
            + str(self.dtype)
            + ", subpath="
            + "'"
            + self.subpath
            + "', slice="
            + str(self.slice_)
            + ")"
        )

    def __iter__(self):
        """Returns Iterable over samples"""
        if isinstance(self.indexes, int):
            yield self
            return

        for i in range(len(self.indexes)):
            yield self[i]

    @property
    def chunksize(self):
        return self.dataset._tensors[self.subpath].chunksize

    @property
    def is_dynamic(self):
        return self.dataset._tensors[self.subpath].is_dynamic

    def disable_lazy(self):
        self.lazy = False

    def enable_lazy(self):
        self.lazy = True
