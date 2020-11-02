import collections.abc as abc
from hub.api.dataset_utils import slice_split


class TensorView:
    def __init__(
        self,
        dataset=None,
        subpath=None,
        slice_=None,
    ):

        assert dataset is not None
        assert subpath is not None

        self.dataset = dataset
        self.subpath = subpath

        if isinstance(slice_, int) or isinstance(slice_, slice):
            self.slice_ = [slice_]
        elif isinstance(slice_, tuple) or isinstance(slice_, list):
            self.slice_ = list(slice_)
        self.nums = []
        self.offsets = []
        for it in self.slice_:
            if isinstance(it, int):
                self.nums.append(1)
                self.offsets.append(it)
            elif isinstance(it, slice):
                ofs = it.start or 0
                num = it.stop - ofs if it.stop else None
                self.nums.append(num)
                self.offsets.append(ofs)

    def numpy(self):
        """Gets the value from tensorview"""
        return self.dataset._tensors[self.subpath][self.slice_]

    def compute(self):
        """Gets the value from tensorview"""
        return self.numpy()

    def __getitem__(self, slice_):
        """Gets a slice or slices from tensorview"""
        if not isinstance(slice_, abc.Iterable) or isinstance(slice_, str):
            slice_ = [slice_]
        slice_ = list(slice_)
        slice_ = [0] + slice_ if self.nums[0] == 1 else slice_
        subpath, slice_list = slice_split(slice_)
        if subpath:
            raise ValueError("Can't slice a dataset with multiple slices without subpath")
        else:
            new_nums = self.nums.copy()
            new_offsets = self.offsets.copy()
            if len(new_nums) < len(slice_list):
                new_nums.extend([None] * (len(slice_list) - len(new_nums)))
                new_offsets.extend([0] * (len(slice_list) - len(new_offsets)))
            for i in range(len(slice_list)):
                slice_list[i] = self._combine(slice_[i], new_nums[i], new_offsets[i])
            for i in range(len(slice_list), len(new_nums)):
                slice_list.append(slice(new_offsets[i], new_offsets[i] + new_nums[i]))
            return TensorView(dataset=self.dataset, subpath=self.subpath, slice_=slice_list)

    def __setitem__(self, slice_, value):
        """"Sets a slice or slices with a value"""
        if not isinstance(slice_, abc.Iterable) or isinstance(slice_, str):
            slice_ = [slice_]
        slice_ = list(slice_)
        slice_ = [0] + slice_ if self.nums[0] == 1 else slice_
        subpath, slice_list = slice_split(slice_, all_slices=False)
        if subpath:
            raise ValueError("Can't slice a dataset with multiple slices without subpath")
        else:
            new_nums = self.nums.copy()
            new_offsets = self.offsets.copy()
            if len(new_nums) < len(slice_list):
                new_nums.extend([None] * (len(slice_list) - len(new_nums)))
                new_offsets.extend([0] * (len(slice_list) - len(new_offsets)))
            for i in range(len(slice_list)):
                slice_list[i] = self._combine(slice_[i], new_nums[i], new_offsets[i])
            for i in range(len(slice_list), len(new_nums)):
                slice_list.append(slice(new_offsets[i], new_offsets[i] + new_nums[i]))
            self.dataset._tensors[self.subpath][slice_list] = value

    def _combine(self, slice_, num=None, ofs=0):
        "combines a slice_ with the current num and offset present in tensorview"
        if isinstance(slice_, int):
            self.check_slice_bounds(num=num, start=slice_)
            return ofs + slice_
        elif isinstance(slice_, slice):
            self.check_slice_bounds(num=num, start=slice_.start, stop=slice_.stop, step=slice_.step)
            if slice_.start is None and slice_.stop is None:
                return slice(ofs, None) if num is None else slice(ofs, ofs + num)
            elif slice_.stop is None:
                return slice(ofs + slice_.start, None) if num is None else slice(ofs + slice_.start, ofs + num)
            elif slice_.start is None:
                return slice(ofs, ofs + slice_.stop)
            else:
                return slice(ofs + slice_.start, ofs + slice_.stop)
        else:
            raise TypeError(
                "type {} isn't supported in dataset slicing".format(type(slice_))
            )

    def check_slice_bounds(self, num=None, start=None, stop=None, step=None):
        "checks whether the bounds of slice are in limits"
        if (step and step < 0):  # negative step not supported
            raise ValueError("Negative step not supported in dataset slicing")
        if num and ((start and start >= num) or (stop and stop > num)):
            raise IndexError(
                "index out of bounds for dimension with length {}".format(
                    num
                )
            )
        if start and stop and start > stop:
            raise IndexError("start index is greater than stop index")
