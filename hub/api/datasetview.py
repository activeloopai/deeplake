import collections.abc as abc
from hub.api.dataset_utils import (
    create_numpy_dict,
    get_value,
    slice_extract_info,
    slice_split,
    str_to_int,
)
from hub.exceptions import NoneValueException
from hub.schema import Sequence, Tensor, SchemaDict, Primitive, Text


class DatasetView:
    def __init__(
        self,
        dataset=None,
        num_samples: int = None,
        offset: int = None,
        squeeze_dim: bool = False,
        lazy: bool = True,
    ):
        """Creates a DatasetView object for a subset of the Dataset.

        Parameters
        ----------
        dataset: hub.api.dataset.Dataset object
            The dataset whose DatasetView is being created
        num_samples: int
            The number of samples in this DatasetView
        offset: int
            The offset from which the DatasetView starts
        squeeze_dim: bool, optional
            For slicing with integers we would love to remove the first dimension to make it nicer
        lazy: bool, optional
            Setting this to False will stop lazy computation and will allow items to be accessed without .compute()
        """
        if dataset is None:
            raise NoneValueException("dataset")
        if num_samples is None:
            raise NoneValueException("num_samples")
        if offset is None:
            raise NoneValueException("offset")

        self.dataset = dataset
        self.num_samples = num_samples
        self.offset = offset
        self.squeeze_dim = squeeze_dim
        self.lazy = lazy

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

        slice_list = [0] + slice_list if self.squeeze_dim else slice_list

        if not subpath:
            if len(slice_list) > 1:
                raise ValueError(
                    "Can't slice a dataset with multiple slices without subpath"
                )
            num, ofs = slice_extract_info(slice_list[0], self.num_samples)
            return DatasetView(
                dataset=self.dataset,
                num_samples=num,
                offset=ofs + self.offset,
                squeeze_dim=isinstance(slice_list[0], int),
                lazy=self.lazy,
            )
        elif not slice_list:
            slice_ = (
                slice(self.offset, self.offset + self.num_samples)
                if not self.squeeze_dim
                else self.offset
            )
            if subpath in self.dataset._tensors.keys():
                tensorview = TensorView(
                    dataset=self.dataset,
                    subpath=subpath,
                    slice_=slice_,
                    lazy=self.lazy,
                )
                return tensorview if self.lazy else tensorview.compute()
            for key in self.dataset._tensors.keys():
                if subpath.startswith(key):
                    objectview = ObjectView(
                        dataset=self.dataset,
                        subpath=subpath,
                        slice_list=[slice_],
                        lazy=self.lazy,
                    )
                    return objectview if self.lazy else objectview.compute()
            return self._get_dictionary(self.dataset, subpath, slice=slice_)
        else:
            num, ofs = slice_extract_info(slice_list[0], self.num_samples)
            slice_list[0] = (
                ofs + self.offset
                if isinstance(slice_list[0], int)
                else slice(ofs + self.offset, ofs + self.offset + num)
            )
            schema_obj = self.dataset.schema.dict_[subpath.split("/")[1]]

            if subpath in self.dataset._tensors.keys() and (
                not isinstance(schema_obj, Sequence) or len(slice_list) <= 1
            ):
                tensorview = TensorView(
                    dataset=self.dataset,
                    subpath=subpath,
                    slice_=slice_list,
                    lazy=self.lazy,
                )
                return tensorview if self.lazy else tensorview.compute()
            for key in self.dataset._tensors.keys():
                if subpath.startswith(key):
                    objectview = ObjectView(
                        dataset=self.dataset,
                        subpath=subpath,
                        slice_list=slice_list,
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
        assign_value = get_value(value)
        # handling strings and bytes
        assign_value = str_to_int(assign_value, self.dataset.tokenizer)

        if not isinstance(slice_, abc.Iterable) or isinstance(slice_, str):
            slice_ = [slice_]
        slice_ = list(slice_)
        subpath, slice_list = slice_split(slice_)
        slice_list = [0] + slice_list if self.squeeze_dim else slice_list
        if not subpath:
            raise ValueError("Can't assign to dataset sliced without subpath")
        elif not slice_list:
            slice_ = (
                self.offset
                # if self.num_samples == 1
                if self.squeeze_dim
                else slice(self.offset, self.offset + self.num_samples)
            )
            if subpath in self.dataset._tensors.keys():
                self.dataset._tensors[subpath][slice_] = assign_value  # Add path check
            for key in self.dataset._tensors.keys():
                if subpath.startswith(key):
                    ObjectView(
                        dataset=self.dataset, subpath=subpath, slice_list=[slice_]
                    )[:] = assign_value
            # raise error
        else:
            num, ofs = (
                slice_extract_info(slice_list[0], self.num_samples)
                if isinstance(slice_list[0], slice)
                else (1, slice_list[0])
            )
            slice_list[0] = (
                slice(ofs + self.offset, ofs + self.offset + num)
                if isinstance(slice_list[0], slice)
                else ofs + self.offset
            )
            # self.dataset._tensors[subpath][slice_list] = assign_value
            if subpath in self.dataset._tensors.keys():
                self.dataset._tensors[subpath][
                    slice_list
                ] = assign_value  # Add path check
                return
            for key in self.dataset._tensors.keys():
                if subpath.startswith(key):
                    ObjectView(
                        dataset=self.dataset, subpath=subpath, slice_list=slice_list
                    )[:] = assign_value

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
        for key in self.dataset._tensors.keys():
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
        if self.squeeze_dim:
            assert len(self) == 1
            yield self
            return

        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return self.num_samples

    def __str__(self):
        out = "DatasetView(" + str(self.dataset) + ", slice="
        out = (
            out + str(self.offset)
            if self.squeeze_dim
            else out + str(slice(self.offset, self.offset + self.num_samples))
        )
        out += ")"
        return out

    def __repr__(self):
        return self.__str__()

    def to_tensorflow(self):
        """Converts the dataset into a tensorflow compatible format"""
        return self.dataset.to_tensorflow(
            num_samples=self.num_samples, offset=self.offset
        )

    def to_pytorch(
        self,
        transform=None,
        inplace=True,
        output_type=dict,
    ):
        """Converts the dataset into a pytorch compatible format"""
        return self.dataset.to_pytorch(
            transform=transform,
            num_samples=self.num_samples,
            offset=self.offset,
            inplace=inplace,
            output_type=output_type,
        )

    def resize_shape(self, size: int) -> None:
        """Resize dataset shape, not DatasetView"""
        self.dataset.resize_shape(size)

    def commit(self) -> None:
        """Commit dataset"""
        self.dataset.commit()

    def numpy(self):
        if self.num_samples == 1 and self.squeeze_dim:
            return create_numpy_dict(self.dataset, self.offset)
        else:
            return [
                create_numpy_dict(self.dataset, self.offset + i)
                for i in range(self.num_samples)
            ]

    def disable_lazy(self):
        self.lazy = False

    def enable_lazy(self):
        self.lazy = True

    def compute(self):
        return self.numpy()


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
        self.nums = []
        self.offsets = []

        self.squeeze_dims = []
        for it in self.slice_:
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
        self.nums[0] = (
            self.dataset.shape[0] - self.offsets[0]
            if self.nums[0] is None
            else self.nums[0]
        )
        self.dtype = self.dtype_from_path(subpath)
        self.shape = self.dataset._tensors[self.subpath].get_shape(self.slice_)

    def numpy(self):
        """Gets the value from tensorview"""
        if isinstance(self.dtype, Text):
            value = self.dataset._tensors[self.subpath][self.slice_]
            if self.dataset.tokenizer is not None:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
                if value.ndim == 1:
                    return tokenizer.decode(value.tolist())
            elif value.ndim == 1:
                return "".join(chr(it) for it in value.tolist())
            raise ValueError("Can only access Text with integer index")
        return self.dataset._tensors[self.subpath][self.slice_]

    def compute(self):
        """Gets the value from tensorview"""
        return self.numpy()

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
        if len(new_nums) < len(slice_list):
            new_nums.extend([None] * (len(slice_list) - len(new_nums)))
            new_offsets.extend([0] * (len(slice_list) - len(new_offsets)))
        for i in range(len(slice_list)):
            slice_list[i] = self._combine(slice_list[i], new_nums[i], new_offsets[i])
        for i in range(len(slice_list), len(new_nums)):
            cur_slice = (
                slice(new_offsets[i], new_offsets[i] + new_nums[i])
                if new_nums[i] > 1
                else new_offsets[i]
            )
            slice_list.append(cur_slice)
        if subpath or (
            len(slice_list) > len(self.nums) and isinstance(self.dtype, Sequence)
        ):
            objectview = ObjectView(
                dataset=self.dataset,
                subpath=self.subpath + subpath,
                slice_list=slice_list,
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
        assign_value = get_value(value)
        # handling strings and bytes
        assign_value = str_to_int(assign_value, self.dataset.tokenizer)

        if not isinstance(slice_, abc.Iterable) or isinstance(slice_, str):
            slice_ = [slice_]
        slice_ = list(slice_)
        slice_ = self.slice_fill(slice_)
        subpath, slice_list = slice_split(slice_)
        new_nums = self.nums.copy()
        new_offsets = self.offsets.copy()
        if len(new_nums) < len(slice_list):
            new_nums.extend([None] * (len(slice_list) - len(new_nums)))
            new_offsets.extend([0] * (len(slice_list) - len(new_offsets)))
        for i in range(len(slice_list)):
            slice_list[i] = self._combine(slice_list[i], new_nums[i], new_offsets[i])
        for i in range(len(slice_list), len(new_nums)):
            cur_slice = (
                slice(new_offsets[i], new_offsets[i] + new_nums[i])
                if new_nums[i] > 1
                else new_offsets[i]
            )
            slice_list.append(cur_slice)
        if subpath or (
            len(slice_list) > len(self.nums) and isinstance(self.dtype, Sequence)
        ):
            ObjectView(
                dataset=self.dataset,
                subpath=self.subpath + subpath,
                slice_list=slice_list,
            )[:] = assign_value
        else:
            self.dataset._tensors[self.subpath][slice_list] = assign_value

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
        new_slice_ = []
        offset = 0
        for i, num in enumerate(self.nums):
            if num == 1 and self.squeeze_dims[i]:
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

    def set_shape(self):
        if self.is_dynamic:
            self.shape = [
                self.dataset._tensors[self.subpath].get_shape([i] + self.slice_[1:])
                for i in range(self.offsets[0], self.offsets[0] + self.nums[0])
            ]
            if len(self.shape) == 1:
                self.shape = self.shape[0]
                self.shape = (
                    (1,) + self.shape
                    if isinstance(self.slice_[0], slice)
                    else self.shape
                )
        else:
            self.shape = self.dataset._tensors[self.subpath].get_shape(self.slice_)

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


class ObjectView:
    def __init__(
        self,
        dataset,
        subpath=None,
        slice_list=None,
        nums=[],
        offsets=[],
        squeeze_dims=[],
        inner_schema_obj=None,
        lazy=True,
        new=True,
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
        inner_schema_obj: Child of hub.schema.Tensor or hub.schema.SchemaDict
            The deepest element in the schema upto which the previous ObjectView had been processed

        new: bool
            Whether to create a new ObjectView object from a Dataset, DatasetView or TensorView
            or create a different ObjectView from an existing one
        """
        self.dataset = dataset
        self.schema = (
            dataset.schema.dict_
            if not isinstance(dataset, DatasetView)
            else dataset.dataset.schema.dict_
        )
        self.subpath = subpath

        self.nums = nums
        self.offsets = offsets
        self.squeeze_dims = squeeze_dims

        self.inner_schema_obj = inner_schema_obj
        self.lazy = lazy

        if new:
            # Creating new obj
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
            # Check if dataset view needs to be made
            if slice_list and len(slice_list) >= 1:
                num, ofs = slice_extract_info(slice_list[0], dataset.shape[0])
                self.dataset = DatasetView(
                    dataset, num, ofs, isinstance(slice_list[0], int)
                )

            if slice_list and len(slice_list) > 1:
                slice_list = slice_list[1:]
                if len(slice_list) > len(self.nums):
                    raise IndexError("Too many indices")
                for i, it in enumerate(slice_list):
                    num, ofs = slice_extract_info(it, self.nums[i])
                    self.nums[i] = num
                    self.offsets[i] += ofs
                    self.squeeze_dims[i] = num == 1

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
        """Checks if a subpath is valid or not. Does not repeat computation done in a
        previous ObjectView object"""
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

        dataset = self.dataset
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
        if len(slice_list) >= 1:
            # Slice first dim
            if isinstance(self.dataset, DatasetView) and not self.dataset.squeeze_dim:
                dataset = self.dataset[slice_list[0]]
                slice_list = slice_list[1:]
            elif not isinstance(self.dataset, DatasetView):
                num, ofs = slice_extract_info(slice_list[0], self.dataset.shape[0])
                dataset = DatasetView(
                    self.dataset, num, ofs, isinstance(slice_list[0], int)
                )
                slice_list = slice_list[1:]

            # Expand slice list for rest of dims
            if len(slice_list) >= 1:
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
                        squeeze_dims[i] = num == 1
        objectview = ObjectView(
            dataset=dataset,
            subpath=subpath,
            slice_list=None,
            nums=nums,
            offsets=offsets,
            squeeze_dims=squeeze_dims,
            inner_schema_obj=inner_schema_obj,
            lazy=self.lazy,
            new=False,
        )
        return objectview if self.lazy else objectview.compute()

    def numpy(self):
        """Gets the value from the objectview"""
        if not isinstance(self.dataset, DatasetView):
            # subpath present but no slice done
            if len(self.subpath.split("/")[1:]) > 1:
                raise IndexError("Can only go deeper on single datapoint")
        if not self.dataset.squeeze_dim:
            # return a combined tensor for multiple datapoints
            # only possible if the field has a fixed size
            paths = self.subpath.split("/")[1:]
            if len(paths) > 1:
                raise IndexError("Can only go deeper on single datapoint")
        else:
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
                    value = self.dataset[paths[0]].compute()
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
                    return self.dataset[paths[0]].compute()[tuple(slice_)]

    def compute(self):
        return self.numpy()

    def __setitem__(self, slice_, value):
        """| Sets a slice of the objectview with a value"""
        if isinstance(slice_, slice) and (slice_.start is None and slice_.stop is None):
            objview = self
        else:
            objview = self.__getitem__(slice_)
        assign_value = get_value(value)

        if not isinstance(objview.dataset, DatasetView):
            # subpath present but no slice done
            assign_value = str_to_int(assign_value, objview.dataset.tokenizer)
            if len(objview.subpath.split("/")[1:]) > 1:
                raise IndexError("Can only go deeper on single datapoint")
        if not objview.dataset.squeeze_dim:
            # assign a combined tensor for multiple datapoints
            # only possible if the field has a fixed size
            assign_value = str_to_int(assign_value, objview.dataset.dataset.tokenizer)
            paths = objview.subpath.split("/")[1:]
            if len(paths) > 1:
                raise IndexError("Can only go deeper on single datapoint")
        else:
            # single datapoint
            def assign(paths, value):
                # helper function for recursive assign
                if len(paths) > 0:
                    path = paths.pop(0)
                    value[path] = assign(paths, value[path])
                    return value
                try:
                    value[tuple(slice_)] = assign_value
                except TypeError:
                    value = assign_value
                return value

            assign_value = str_to_int(assign_value, objview.dataset.dataset.tokenizer)
            paths = objview.subpath.split("/")[1:]
            schema = objview.schema[paths[0]]
            slice_ = [
                of if sq else slice(of, of + num) if num else slice(None, None)
                for num, of, sq in zip(
                    objview.nums, objview.offsets, objview.squeeze_dims
                )
            ]
            if isinstance(schema, Sequence):
                if isinstance(schema.dtype, SchemaDict):
                    # if sequence of dict, have to fetch everything
                    value = objview.dataset[paths[0]].compute()
                    value = assign(paths[1:], value)
                    objview.dataset[paths[0]] = value
                else:
                    # sequence of tensors
                    value = objview.dataset[paths[0]].compute()
                    value[tuple(slice_)] = assign_value
                    objview.dataset[paths[0]] = value

    def __str__(self):
        if isinstance(self.dataset, DatasetView):
            slice_ = [
                self.dataset.offset
                if self.dataset.squeeze_dim
                else slice(
                    self.dataset.offset, self.dataset.offset + self.dataset.num_samples
                )
            ]
        else:
            slice_ = [slice(None, None)]
        slice_ += [
            ofs if sq else slice(ofs, ofs + num) if num else slice(None, None)
            for ofs, num, sq in zip(self.offsets, self.nums, self.squeeze_dims)
        ]
        return f"ObjectView(subpath='{self.subpath}', slice={str(slice_)})"
