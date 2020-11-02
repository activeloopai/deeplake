from hub.api.tensorview import TensorView
from hub.api.dataset_utils import slice_extract_info, slice_split
import collections.abc as abc


class DatasetView:
    def __init__(self, dataset=None, num_samples=None, offset=None):
        assert dataset is not None
        assert num_samples is not None
        assert offset is not None

        self.dataset = dataset
        self.num_samples = num_samples
        self.offset = offset

    def __getitem__(self, slice_):
        """Gets a slice or slices from datasetview"""
        if not isinstance(slice_, abc.Iterable) or isinstance(slice_, str):
            slice_ = [slice_]
        slice_ = list(slice_)
        subpath, slice_list = slice_split(slice_)
        slice_list = [0] + slice_list if self.num_samples == 1 else slice_list
        if not subpath:
            if len(slice_list) > 1:
                raise ValueError("Can't slice a dataset with multiple slices without subpath")
            num, ofs = slice_extract_info(slice_list[0], self.num_samples)
            return DatasetView(dataset=self.dataset, num_samples=num, offset=ofs + self.offset)
        elif not slice_list:
            slice_ = slice(self.offset, self.offset + self.num_samples)
            if subpath in self.dataset._tensors.keys():
                return TensorView(dataset=self.dataset, subpath=subpath, slice_=slice_)
            return self._get_dictionary(self.dataset, subpath, slice=slice_)
        else:
            num, ofs = slice_extract_info(slice_list[0], self.num_samples) if isinstance(slice_list[0], slice) else (1, slice_list[0])
            slice_list[0] = ofs + self.offset if num == 1 else slice(ofs + self.offset, ofs + self.offset + num)
            if subpath in self.dataset._tensors.keys():
                return TensorView(dataset=self.dataset, subpath=subpath, slice_=slice_list)
            if len(slice_list) > 1:
                raise ValueError("You can't slice a dictionary of Tensors")
            return self._get_dictionary(subpath, slice_list[0])

    def __setitem__(self, slice_, value):
        """"Sets a slice or slices with a value"""
        if not isinstance(slice_, abc.Iterable) or isinstance(slice_, str):
            slice_ = [slice_]
        slice_ = list(slice_)
        subpath, slice_list = slice_split(slice_, all_slices=False)
        slice_list = [0] + slice_list if self.num_samples == 1 else slice_list
        if not subpath:
            raise ValueError("Can't assign to dataset sliced without subpath")
        elif not slice_list:
            slice_ = self.offset if self.num_samples == 1 else slice(self.offset, self.offset + self.num_samples)
            self.dataset._tensors[subpath][slice_] = value  # Add path check
        else:
            num, ofs = slice_extract_info(slice_list[0], self.num_samples) if isinstance(slice_list[0], slice) else (1, slice_list[0])
            slice_list[0] = slice(ofs + self.offset, ofs + self.offset + num) if num > 1 else ofs + self.offset
            self.dataset._tensors[subpath][slice_list] = value

    def _get_dictionary(self, subpath, slice_=None):
        """"Gets dictionary from dataset given incomplete subpath"""
        tensor_dict = {}
        subpath = subpath if subpath.endswith("/") else subpath + "/"
        for key in self.dataset._tensors.keys():
            if key.startswith(subpath):
                suffix_key = key[len(subpath) :]
                split_key = suffix_key.split("/")
                cur = tensor_dict
                for i in range(len(split_key) - 1):
                    if split_key[i] not in cur.keys():
                        cur[split_key[i]] = {}
                    cur = cur[split_key[i]]
                slice_ = slice_ or slice(0, self.dataset.shape[0])
                cur[split_key[-1]] = TensorView(
                    dataset=self.dataset, subpath=key, slice_=slice_
                )
        if not tensor_dict:
            raise KeyError(f"Key {subpath} was not found in dataset")
        return tensor_dict
