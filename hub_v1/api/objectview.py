"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from hub_v1.schema import Sequence, Tensor, SchemaDict, Primitive
from hub_v1.api.dataset_utils import slice_extract_info, slice_split

import collections.abc as abc


class ObjectView:
    def __init__(
        self,
        dataset,
        subpath=None,
        slice_=None,
        indexes=None,
        nums=[],
        offsets=[],
        squeeze_dims=[],
        inner_schema_obj=None,
        lazy=True,
        check_bounds=True,
    ):
        """Creates an ObjectView object for dataset from a Dataset, DatasetView or TensorView
        object, or creates a different ObjectView from an existing one

        Parameters
        ----------
        These parameters are used to create a new ObjectView.
        dataset: hub.api.dataset.Dataset object
            The dataset whose ObjectView is being created, or its DatasetView
        subpath: str (optional)
            A potentially incomplete path to any element in the Dataset
        slice_list: optional
            The `slice_` of this Tensor that needs to be accessed
        lazy: bool, optional
            Setting this to False will stop lazy computation and will allow items to be accessed without .compute()

        These parameters are also needed to create an ObjectView from an existing one.
        nums: List[int]
            Number of elements in each dimension of the ObjectView to be created
        offsets: List[int]
            Starting element in each dimension of the ObjectView to be created
        squeeze_dims: List[bool]
            Whether each dimension can be squeezed or not
        inner_schema_obj: Child of hub_v1.schema.Tensor or hub_v1.schema.SchemaDict
            The deepest element in the schema upto which the previous ObjectView had been processed
        check_bounds: bool
            Whether to create a new ObjectView object from a Dataset, DatasetView or TensorView
            or create a different ObjectView from an existing one
        """
        self.dataset = dataset
        self.schema = dataset.schema.dict_
        self.subpath = subpath

        self.nums = nums
        self.offsets = offsets
        self.squeeze_dims = squeeze_dims

        self.inner_schema_obj = inner_schema_obj
        self.lazy = lazy

        if check_bounds:
            if self.subpath:
                (
                    self.inner_schema_obj,
                    self.nums,
                    self.offsets,
                    self.squeeze_dims,
                ) = self.process_path(
                    self.subpath,
                    self.inner_schema_obj,
                    self.nums.copy(),
                    self.offsets.copy(),
                    self.squeeze_dims.copy(),
                )
            if slice_ and len(slice_) >= 1:
                self.indexes = slice_[0]
                self.is_contiguous = False
                if isinstance(self.indexes, list) and self.indexes:
                    self.is_contiguous = self.indexes[-1] - self.indexes[0] + 1 == len(
                        self.indexes
                    )
                slice_ = slice_[1:]
                if len(slice_) > len(self.nums):
                    raise IndexError("Too many indices")
                for i, it in enumerate(slice_):
                    num, ofs = slice_extract_info(it, self.nums[i])
                    self.nums[i] = num
                    self.offsets[i] += ofs
                    self.squeeze_dims[i] = num == 1
        else:
            self.indexes = indexes
            self.is_contiguous = False
            if isinstance(self.indexes, list) and self.indexes:
                self.is_contiguous = self.indexes[-1] - self.indexes[0] + 1 == len(
                    self.indexes
                )

    def num_process(self, schema_obj, nums, offsets, squeeze_dims):
        """Determines the maximum number of elements in each discovered dimension"""
        if isinstance(schema_obj, SchemaDict):
            return
        elif isinstance(schema_obj, Sequence):
            nums.append(0)
            offsets.append(0)
            squeeze_dims.append(False)
            if isinstance(schema_obj.dtype, Tensor):
                self.num_process(schema_obj.dtype, nums, offsets, squeeze_dims)
        else:
            for dim in schema_obj.max_shape:
                nums.append(dim)
                offsets.append(0)
                squeeze_dims.append(False)
        if not isinstance(schema_obj.dtype, Primitive) and not isinstance(
            schema_obj, Sequence
        ):
            raise ValueError("Only sequences can be nested")

    def process_path(self, subpath, inner_schema_obj, nums, offsets, squeeze_dims):
        """Checks if a subpath is valid or not. Does not repeat computation done in a previous ObjectView object"""
        paths = subpath.split("/")[1:]
        try:
            # If key is invalid raises KeyError
            # If schema object is not subscriptable raises AttributeError
            if inner_schema_obj:
                if isinstance(inner_schema_obj, Sequence):
                    schema_obj = inner_schema_obj.dtype.dict_[paths[0]]
                elif isinstance(inner_schema_obj, SchemaDict):
                    schema_obj = inner_schema_obj.dict_[paths[0]]
                else:
                    raise KeyError()
            else:
                schema_obj = self.schema[paths[0]]
        except (KeyError, AttributeError):
            raise KeyError(f"{paths[0]} is an invalid key")
        self.num_process(schema_obj, nums, offsets, squeeze_dims)
        for path in paths[1:]:
            try:
                if isinstance(schema_obj, Sequence):
                    schema_obj = schema_obj.dtype.dict_[path]
                elif isinstance(schema_obj, SchemaDict):
                    schema_obj = schema_obj.dict_[path]
                else:
                    raise KeyError()
                self.num_process(schema_obj, nums, offsets, squeeze_dims)
            except (KeyError, AttributeError):
                raise KeyError(f"{path} is an invalid key")
        return schema_obj, nums, offsets, squeeze_dims

    def __getitem__(self, slice_):
        """| Gets a slice from an objectview"""
        if not isinstance(slice_, abc.Iterable) or isinstance(slice_, str):
            slice_ = [slice_]
        slice_ = list(slice_)
        subpath, slice_list = slice_split(slice_)

        nums, offsets, squeeze_dims, inner_schema_obj = (
            self.nums.copy(),
            self.offsets.copy(),
            self.squeeze_dims.copy(),
            self.inner_schema_obj,
        )

        if subpath:
            inner_schema_obj, nums, offsets, squeeze_dims = self.process_path(
                subpath, inner_schema_obj, nums, offsets, squeeze_dims
            )
        subpath = self.subpath + subpath

        new_indexes = self.indexes
        if len(slice_list) >= 1:
            if isinstance(self.indexes, list):
                new_indexes = self.indexes[slice_list[0]]
                if self.is_contiguous and new_indexes:
                    new_indexes = slice(new_indexes[0], new_indexes[-1] + 1)
                slice_list = slice_list[1:]
            elif isinstance(self.indexes, slice):
                ofs = self.indexes.start or 0
                num = self.indexes.stop - ofs if self.indexes.stop else None
                num, ofs_temp = slice_extract_info(slice_list[0], num)
                new_indexes = (
                    ofs + ofs_temp
                    if isinstance(slice_list[0], int)
                    else slice(ofs + ofs_temp, ofs + ofs_temp + num)
                )
                slice_list = slice_list[1:]

        if len(slice_list) >= 1:
            # Expand slice list
            exp_slice_list = []
            for squeeze in squeeze_dims:
                if squeeze:
                    exp_slice_list += [None]
                else:
                    if len(slice_list) > 0:
                        exp_slice_list += [slice_list.pop(0)]
                    else:
                        # slice list smaller than max
                        exp_slice_list += [None]
            if len(slice_list) > 0:
                # slice list longer than max
                raise IndexError("Too many indices")
            for i, it in enumerate(exp_slice_list):
                if it is not None:
                    num, ofs = slice_extract_info(it, nums[i])
                    nums[i] = num
                    offsets[i] += ofs
                    squeeze_dims[i] = isinstance(it, int)

        objectview = ObjectView(
            dataset=self.dataset,
            subpath=subpath,
            slice_=None,
            indexes=new_indexes,
            nums=nums,
            offsets=offsets,
            squeeze_dims=squeeze_dims,
            inner_schema_obj=inner_schema_obj,
            lazy=self.lazy,
            check_bounds=False,
        )
        return objectview if self.lazy else objectview.compute()

    def numpy(self):
        """Gets the value from the objectview"""
        if isinstance(self.indexes, list):
            if len(self.indexes) > 1:
                raise IndexError("Can only go deeper on single datapoint")
            else:
                slice_0 = self.indexes[0]
        elif isinstance(self.indexes, slice):
            if self.indexes.stop - self.indexes.start > 1:
                raise IndexError("Can only go deeper on single datapoint")
            slice_0 = self.indexes.start
        else:
            slice_0 = self.indexes
        # single datapoint
        paths = self.subpath.split("/")[1:]
        schema = self.schema[paths[0]]
        slice_ = [
            ofs if sq else slice(ofs, ofs + num) if num else slice(None, None)
            for ofs, num, sq in zip(self.offsets, self.nums, self.squeeze_dims)
        ]
        if isinstance(schema, Sequence):
            if isinstance(schema.dtype, SchemaDict):
                # if sequence of dict, have to fetch everything
                lazy = self.dataset.lazy
                self.dataset.lazy = False
                value = self.dataset[[paths[0], slice_0]]
                self.dataset.lazy = lazy
                for path in paths[1:]:
                    value = value[path]
                try:
                    return value[tuple(slice_)]
                except TypeError:
                    # raise error
                    return value
                except KeyError:
                    raise KeyError("Invalid slice")
            else:
                # sequence of tensors
                return self.dataset[[paths[0], slice_0]].compute()[tuple(slice_)]

    def compute(self):
        return self.numpy()

    def __str__(self):
        slice_ = [
            ofs if sq else slice(ofs, ofs + num) if num else slice(None, None)
            for ofs, num, sq in zip(self.offsets, self.nums, self.squeeze_dims)
        ]
        return f"ObjectView(subpath='{self.subpath}', indexes={str(self.indexes)}, slice={str(slice_)})"
