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
                if it.start is None:
                    ofs = 0
                else:
                    ofs = it.start

                if it.stop is None:
                    num = None
                else:
                    num = it.stop - ofs
                # num, ofs = slice_extract_info(it)
                self.nums.append(num)
                self.offsets.append(ofs)

    # TODO Add support for incomplete paths

    def numpy(self):
        return self.dataset._tensors[self.subpath][self.slice_]

    # TODO Add slicing logic to tensorview
    def __getitem__(self, slice_):
        if isinstance(slice_, int):
            slice_ = self._combine(slice_, self.nums[0], self.offsets[0])
            slice_ = [slice_]
        elif isinstance(slice_, slice):
            slice_ = self._combine(slice_, self.nums[0], self.offsets[0])
            slice_ = [slice_]
        elif isinstance(slice_, tuple):
            new_nums = self.nums
            new_offsets = self.offsets
            slice_ = list(slice_)
            if len(new_nums) < len(slice_):
                new_nums.extend([None] * (len(slice_) - len(new_nums)))
                new_offsets.extend([0] * (len(slice_) - len(new_offsets)))
            for i in range(len(slice_)):
                slice_[i] = self._combine(slice_[i], new_nums[i], new_offsets[i])
        else:
            raise TypeError("type {} isn't supported in dataset slicing".format(type(slice_)))

        if len(self.nums) > len(slice_):
            slice_ += self.slice_[len(slice_) - len(self.nums):]
        return TensorView(dataset=self.dataset, subpath=self.subpath, slice_=slice_)

    def __setitem__(self, slice_, value):
        if isinstance(slice_, int):
            slice_ = self._combine(slice_, self.nums[0], self.offsets[0])
        elif isinstance(slice_, slice):
            slice_ = self._combine(slice_, self.nums[0], self.offsets[0])
        elif isinstance(slice_, tuple):
            new_nums = self.nums
            new_offsets = self.offsets
            slice_ = list(slice_)
            if len(new_nums) < len(slice_):
                new_nums.extend([None] * (len(slice_) - len(new_nums)))
                new_offsets.extend([0] * (len(slice_) - len(new_offsets)))
            for i in range(len(slice_)):
                slice_[i] = self._combine(slice_[i], new_nums[i], new_offsets[i])
            if len(new_nums) > len(slice_):
                slice_ += self.new_nums[len(slice_) - len(new_nums):]
        else:
            raise TypeError("type {} isn't supported in dataset slicing".format(type(slice_)))
        self.dataset._tensors[self.subpath][slice_] = value

    def _combine(self, slice_, num=None, ofs=0):
        if isinstance(slice_, int):
            if num is not None and slice_ >= num:
                raise IndexError('index out of bounds for dimension with length {}'.format(num))
            return ofs + slice_
        elif isinstance(slice_, slice):
            if slice_.step is not None and slice_.step < 0:       # negative step not supported
                raise ValueError("Negative step not supported in dataset slicing")
            if slice_.start is None and slice_.stop is None:
                if num is None:
                    return slice(ofs, None)
                else:
                    return slice(ofs, ofs + num)
            elif slice_.start is not None and slice_.stop is None:
                if num is None:
                    return slice(ofs + slice_.start, None)
                else:
                    if slice_.start >= num:
                        raise IndexError('index out of bounds for dimension with length {}'.format(num))
                    return slice(ofs + slice_.start, ofs + num)
            elif slice_.start is None and slice_.stop is not None:
                if num is None:
                    return slice(ofs, ofs + slice_.stop)
                else:
                    if slice_.stop > num:
                        raise IndexError('index out of bounds for dimension with length {}'.format(num))
                    return slice(ofs, ofs + slice_.stop)
            else:
                if slice_.start > slice_.stop:
                    raise IndexError('start index is greater than stop index')
                if num is None:
                    return slice(ofs + slice_.start, ofs + slice_.stop)
                else:
                    if slice_.stop > num:
                        raise IndexError('index out of bounds for dimension with length {}'.format(num))
                    return slice(ofs + slice_.start, ofs + slice_.stop)
        else:
            raise TypeError("type {} isn't supported in dataset slicing".format(type(slice_)))
