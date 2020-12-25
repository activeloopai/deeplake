from hub.api.datasetview import DatasetView
from hub.schema import Sequence, Tensor, SchemaDict, Primitive
from hub.api.dataset_utils import get_value, slice_extract_info, slice_split, str_to_int

# from hub.exceptions import NoneValueException
import collections.abc as abc
import hub.api as api


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
