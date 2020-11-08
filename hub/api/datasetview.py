from hub.api.tensorview import TensorView
from hub.api.dataset_utils import slice_extract_info, slice_split
import collections.abc as abc


class DatasetView:
    def __init__(
        self,
        dataset=None,
        num_samples=None,
        offset=None
    ):
        """Creates a DatasetView object for a subset of the Dataset

        Parameters
        ----------
        dataset: hub.api.dataset.Dataset object
            The dataset whose DatasetView is being created
        num_samples: int
            The number of samples in this DatasetView
        offset: int
            The offset from which the DatasetView starts
        """
        assert dataset is not None
        assert num_samples is not None
        assert offset is not None

        self.dataset = dataset
        self.num_samples = num_samples
        self.offset = offset

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
        slice_list = [0] + slice_list if self.num_samples == 1 else slice_list
        if not subpath:
            if len(slice_list) > 1:
                raise ValueError(
                    "Can't slice a dataset with multiple slices without subpath"
                )
            num, ofs = slice_extract_info(slice_list[0], self.num_samples)
            return DatasetView(
                dataset=self.dataset, num_samples=num, offset=ofs + self.offset
            )
        elif not slice_list:
            slice_ = slice(self.offset, self.offset + self.num_samples)
            if subpath in self.dataset._tensors.keys():
                return TensorView(dataset=self.dataset, subpath=subpath, slice_=slice_)
            return self._get_dictionary(self.dataset, subpath, slice=slice_)
        else:
            num, ofs = slice_extract_info(slice_list[0], self.num_samples)
            slice_list[0] = (
                ofs + self.offset
                if num == 1
                else slice(ofs + self.offset, ofs + self.offset + num)
            )
            if subpath in self.dataset._tensors.keys():
                return TensorView(
                    dataset=self.dataset, subpath=subpath, slice_=slice_list
                )
            if len(slice_list) > 1:
                raise ValueError("You can't slice a dictionary of Tensors")
            return self._get_dictionary(subpath, slice_list[0])

    def __setitem__(self, slice_, value):
        """| Sets a slice or slices with a value
        | Usage:

        >>> ds_view = ds[5:15]
        >>> ds_view["image", 3, 0:1920, 0:1080, 0:3] = np.zeros((1920, 1080, 3), "uint8") # sets the 8th image
        """
        if not isinstance(slice_, abc.Iterable) or isinstance(slice_, str):
            slice_ = [slice_]
        slice_ = list(slice_)
        subpath, slice_list = slice_split(slice_)
        slice_list = [0] + slice_list if self.num_samples == 1 else slice_list
        if not subpath:
            raise ValueError("Can't assign to dataset sliced without subpath")
        elif not slice_list:
            slice_ = (
                self.offset
                if self.num_samples == 1
                else slice(self.offset, self.offset + self.num_samples)
            )
            self.dataset._tensors[subpath][slice_] = value  # Add path check
        else:
            num, ofs = (
                slice_extract_info(slice_list[0], self.num_samples)
                if isinstance(slice_list[0], slice)
                else (1, slice_list[0])
            )
            slice_list[0] = (
                slice(ofs + self.offset, ofs + self.offset + num)
                if num > 1
                else ofs + self.offset
            )
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
                slice_ = slice_ if slice_ else slice(0, self.dataset.shape[0])
                cur[split_key[-1]] = TensorView(
                    dataset=self.dataset, subpath=key, slice_=slice_
                )
        if len(tensor_dict) == 0:
            raise KeyError(f"Key {subpath} was not found in dataset")
        return tensor_dict
