from hub.api.tensorview import TensorView
from hub.api.dataset_utils import (
    create_numpy_dict,
    get_value,
    slice_extract_info,
    slice_split,
    str_to_int,
)
from hub.exceptions import NoneValueException
import collections.abc as abc
import hub.api.objectview as objv


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
                    objectview = objv.ObjectView(
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
                not isinstance(schema_obj, objv.Sequence) or len(slice_list) <= 1
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
                    objectview = objv.ObjectView(
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
                    objv.ObjectView(
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
                    objv.ObjectView(
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
