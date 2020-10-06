from hub.api.tensorview import TensorView
from hub.api.dataset_utils import slice_extract_info, slice_split_tuple


class DatasetView:
    def __init__(
        self,
        dataset=None,
        num_samples=None,
        offset=None
    ):
        assert dataset is not None
        assert num_samples is not None
        assert offset is not None

        self.dataset = dataset
        self.num_samples = num_samples
        self.offset = offset

    def __getitem__(self, slice_):
        if isinstance(slice_, int):             # return Dataset with single sample
            # doesn't handle negative right now
            if slice_ >= self.num_samples:
                raise IndexError('index out of bounds for dimension with length {}'.format(self.num_samples))
            return DatasetView(dataset=self.dataset, num_samples=1, offset=self.offset + slice_)

        elif isinstance(slice_, slice):         # return Dataset with multiple samples
            num, ofs = slice_extract_info(slice_, self.num_samples)
            return DatasetView(dataset=self.dataset, num_samples=num, offset=self.offset + ofs)

        elif isinstance(slice_, str):
            subpath = slice_ if slice_.startswith("/") else "/" + slice_     # return TensorView object
            # add slice for original Dataset range
            return TensorView(dataset=self.dataset, subpath=subpath, slice_=slice(self.offset, self.offset + self.num_samples))

        elif isinstance(slice_, tuple):        # return tensor view object
            subpath, slice_ = slice_split_tuple(slice_)

            if len(slice_) == 0:
                slice_ = None
            elif isinstance(slice_[0], int):
                if slice_[0] >= self.num_samples:
                    raise IndexError('index out of bounds for dimension with length {}'.format(self.num_samples))
                ls = list(slice_)
                ls[0] += self.offset
                slice_ = tuple(ls)
            elif isinstance(slice_[0], slice):
                num, ofs = slice_extract_info(slice_[0], self.num_samples)
                ls = list(slice_)
                ls[0] = slice(self.offset + ofs, self.offset + ofs + num)
                slice_ = tuple(ls)
            return TensorView(dataset=self.dataset, subpath=subpath, slice_=slice_)
        else:
            raise TypeError("type {} isn't supported in dataset slicing".format(type(slice_)))

    def __setitem__(self, slice_, value):
        if isinstance(slice_, int):             # Not supported
            raise TypeError("Can't assign to dataset indexed only with int")

        elif isinstance(slice_, slice):         # Not supported
            raise TypeError("Can't assign to dataset indexed only with slice")

        elif isinstance(slice_, str):
            slice_ = slice_ if slice_.startswith("/") else "/" + slice_     # return TensorView object
            self.dataset._tensors[slice_][:] = value

        elif isinstance(slice_, tuple):        # return tensor view object
            subpath, slice_ = slice_split_tuple(slice_)
            if len(slice_) == 0:
                slice_ = slice(0, self.num_samples)
            elif isinstance(slice_[0], int):
                offset_slice = slice_[0] + self.offset
                ls = list(slice_)
                ls[0] = offset_slice
                slice_ = tuple(ls)
            elif isinstance(slice_[0], slice):
                offset_slice = slice(slice_[0].start + self.offset, slice_[0].stop + self.offset)
                ls = list(slice_)
                ls[0] = offset_slice
                slice_ = tuple(ls)
            self.dataset._tensors[subpath][slice_] = value
        else:
            raise TypeError("type {} isn't supported in dataset slicing".format(type(slice_)))
