import hub
import collections.abc as abc
from hub.api.dataset_utils import slice_split, str_to_int
from hub.exceptions import NoneValueException


class TensorView:
    def __init__(
        self,
        dataset=None,
        subpath=None,
        slice_=None,
        squeeze_dims=[],
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
        """

        if dataset is None:
            raise NoneValueException("dataset")
        if subpath is None:
            raise NoneValueException("subpath")

        self.dataset = dataset
        self.subpath = subpath

        if isinstance(slice_, int) or isinstance(slice_, slice):
            self.slice_ = [slice_]
        elif isinstance(slice_, tuple) or isinstance(slice_, list):
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
                ofs = it.start if it.start else 0
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
        if isinstance(self.dtype, hub.schema.text.Text):
            value = self.dataset._tensors[self.subpath][self.slice_]
            if self.dataset.tokenizer is not None:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
                if value.ndim == 1:
                    return tokenizer.decode(value.tolist())
            else:
                if value.ndim == 1:
                    return "".join(chr(it) for it in value.tolist())
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

        if subpath:
            raise ValueError("Can't slice a Tensor with string")
        else:
            new_nums = self.nums.copy()
            new_offsets = self.offsets.copy()
            if len(new_nums) < len(slice_list):
                new_nums.extend([None] * (len(slice_list) - len(new_nums)))
                new_offsets.extend([0] * (len(slice_list) - len(new_offsets)))
            for i in range(len(slice_list)):
                slice_list[i] = self._combine(
                    slice_list[i], new_nums[i], new_offsets[i]
                )
            for i in range(len(slice_list), len(new_nums)):
                cur_slice = (
                    slice(new_offsets[i], new_offsets[i] + new_nums[i])
                    if new_nums[i] > 1
                    else new_offsets[i]
                )
                slice_list.append(cur_slice)
            return TensorView(
                dataset=self.dataset, subpath=self.subpath, slice_=slice_list
            )

    def __setitem__(self, slice_, value):
        """| Sets a slice or slices with a value
        | Usage:

        >>> images_tensorview = ds["image"]
        >>> images_tensorview[7, 0:1920, 0:1080, 0:3] = np.zeros((1920, 1080, 3), "uint8") # sets 7th image
        """
        # handling strings and bytes
        assign_value = value
        assign_value = str_to_int(assign_value, self.dataset.tokenizer)

        if not isinstance(slice_, abc.Iterable) or isinstance(slice_, str):
            slice_ = [slice_]
        slice_ = list(slice_)
        slice_ = self.slice_fill(slice_)
        subpath, slice_list = slice_split(slice_)

        if subpath:
            raise ValueError(
                "Can't slice a Tensor with multiple slices without subpath"
            )
        else:
            new_nums = self.nums.copy()
            new_offsets = self.offsets.copy()
            if len(new_nums) < len(slice_list):
                new_nums.extend([None] * (len(slice_list) - len(new_nums)))
                new_offsets.extend([0] * (len(slice_list) - len(new_offsets)))
            for i in range(len(slice_list)):
                slice_list[i] = self._combine(slice_[i], new_nums[i], new_offsets[i])
            for i in range(len(slice_list), len(new_nums)):
                cur_slice = (
                    slice(new_offsets[i], new_offsets[i] + new_nums[i])
                    if new_nums[i] > 1
                    else new_offsets[i]
                )
                slice_list.append(cur_slice)
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
            elif slice_.start is not None and slice_.stop is None:
                return (
                    slice(ofs + slice_.start, None)
                    if num is None
                    else slice(ofs + slice_.start, ofs + num)
                )
            elif slice_.start is None and slice_.stop is not None:
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
        new_slice_ = new_slice_ + slice_[offset:]
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
