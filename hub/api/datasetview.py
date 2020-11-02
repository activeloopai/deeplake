from hub.api.tensorview import TensorView
from hub.api.dataset_utils import slice_extract_info, slice_split_tuple


class DatasetView:
    def __init__(self, dataset=None, num_samples=None, offset=None):
        assert dataset is not None
        assert num_samples is not None
        assert offset is not None

        self.dataset = dataset
        self.num_samples = num_samples
        self.offset = offset

    def __getitem__(self, slice_):
        if self.num_samples == 1:
            if isinstance(slice_, tuple) or isinstance(slice_, list):
                slice_ = tuple(slice_)
                slice_ = (0,) + slice_
            else:
                slice_ = (0, slice_)

        if isinstance(slice_, int):
            # return Dataset with single sample
            # doesn't handle negative right now
            if slice_ >= self.num_samples:
                raise IndexError(
                    "index out of bounds for dimension with length {}".format(
                        self.num_samples
                    )
                )
            return DatasetView(
                dataset=self.dataset, num_samples=1, offset=self.offset + slice_
            )

        elif isinstance(slice_, slice):
            # return Dataset with multiple samples
            num, ofs = slice_extract_info(slice_, self.num_samples)
            return DatasetView(
                dataset=self.dataset, num_samples=num, offset=self.offset + ofs
            )

        elif isinstance(slice_, str):
            subpath = slice_ if slice_.startswith("/") else "/" + slice_
            # return TensorView object
            if self.num_samples == 1:
                slice_ = self.offset
            else:
                slice_ = slice(self.offset, self.offset + self.num_samples)

            if subpath in self._tensors.keys():
                return TensorView(dataset=self.dataset, subpath=subpath, slice_=slice_)
            else:
                d = {}
                post_subpath = subpath if subpath.endswith("/") else subpath + "/"
                for key in self._tensors.keys():
                    if key.startswith(post_subpath):
                        suffix_key = key[len(post_subpath) :]
                    else:
                        continue
                    split_key = suffix_key.split("/")
                    cur = d
                    for i in range(len(split_key) - 1):
                        if split_key[i] not in cur.keys():
                            cur[split_key[i]] = {}
                        cur = cur[split_key[i]]
                    cur[split_key[-1]] = TensorView(
                        dataset=self, subpath=key, slice_=slice_
                    )

                if not d:
                    raise KeyError(f"Key {subpath} was not found in dataset")
                return d

        elif isinstance(slice_, tuple):
            # return tensor view object
            subpath, slice_ = slice_split_tuple(slice_)

            if len(slice_) == 0:
                slice_ = (0, ) if self.num_samples == 1 else (slice(0, self.num_samples), )
            d = {}
            if subpath not in self.dataset._tensors.keys():
                post_subpath = subpath if subpath.endswith("/") else subpath + "/"
                slice_ = list(slice_)
                slice_[0] += self.offset
                for key in self.dataset._tensors.keys():
                    if key.startswith(post_subpath):
                        suffix_key = key[len(post_subpath) :]
                    else:
                        continue
                    split_key = suffix_key.split("/")
                    cur = d
                    for i in range(len(split_key) - 1):
                        if split_key[i] not in cur.keys():
                            cur[split_key[i]] = {}
                        cur = cur[split_key[i]]
                    cur[split_key[-1]] = TensorView(
                        dataset=self.dataset, subpath=key, slice_=slice_
                    )
                if not d:
                    raise KeyError(f"Key {subpath} was not found in dataset")

            if len(slice_) <= 1:
                if len(slice_) == 1:
                    if isinstance(slice_[0], int) and slice_[0] >= self.num_samples:
                        raise IndexError(
                            "index out of bounds for dimension with length {}".format(
                                self.num_samples
                            )
                        )
                    elif isinstance(slice_[0], slice):
                        # will check slice limits and raise error if required
                        num, ofs = slice_extract_info(slice_[0], self.num_samples)
                if subpath in self.dataset._tensors.keys():
                    slice_ = list(slice_)
                    slice_[0] += self.offset
                    return TensorView(
                        dataset=self.dataset, subpath=subpath, slice_=slice_
                    )
                else:
                    return d
            else:
                if subpath not in self.dataset._tensors.keys():
                    raise ValueError("You can't slice a dictionary of Tensors")
                elif isinstance(slice_[0], int):
                    if slice_[0] >= self.num_samples:
                        raise IndexError(
                            "index out of bounds for dimension with length {}".format(
                                self.num_samples
                            )
                        )
                    slice_ = list(slice_)
                    slice_[0] += self.offset
                elif isinstance(slice_[0], slice):
                    num, ofs = slice_extract_info(slice_[0], self.num_samples)
                    slice_ = list(slice_)
                    if num == 1:
                        slice_[0] = self.offset + ofs
                    else:
                        slice_[0] = slice(self.offset + ofs, self.offset + ofs + num)
                return TensorView(dataset=self.dataset, subpath=subpath, slice_=slice_)
        else:
            raise TypeError(
                "type {} isn't supported in dataset slicing".format(type(slice_))
            )

    def __setitem__(self, slice_, value):
        if self.num_samples == 1:
            if isinstance(slice_, tuple) or isinstance(slice_, list):
                slice_ = tuple(slice_)
                slice_ = (0,) + slice_
            else:
                slice_ = (0, slice_)

        if isinstance(slice_, int):  # Not supported
            raise TypeError("Can't assign to dataset indexed only with int")

        elif isinstance(slice_, slice):  # Not supported
            raise TypeError("Can't assign to dataset indexed only with slice")

        elif isinstance(slice_, str):
            slice_ = (
                slice_ if slice_.startswith("/") else "/" + slice_
            )  # return TensorView object
            self.dataset._tensors[slice_][:] = value

        elif isinstance(slice_, tuple):  # return tensor view object
            subpath, slice_ = slice_split_tuple(slice_)
            if len(slice_) == 0:
                slice_ = slice(0, self.num_samples)
            elif isinstance(slice_[0], int):
                offset_slice = slice_[0] + self.offset
                ls = list(slice_)
                ls[0] = offset_slice
                slice_ = tuple(ls)
            elif isinstance(slice_[0], slice):
                offset_slice = slice(
                    slice_[0].start + self.offset, slice_[0].stop + self.offset
                )
                ls = list(slice_)
                ls[0] = offset_slice
                slice_ = tuple(ls)
            self.dataset._tensors[subpath][slice_] = value
        else:
            raise TypeError(
                "type {} isn't supported in dataset slicing".format(type(slice_))
            )
