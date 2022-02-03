from hub.core.transform.transform_tensor import TransformTensor
from hub.util.exceptions import TensorDoesNotExistError


class TransformDataset:
    def __init__(self, all_tensors=None, slice_list=None):
        """Creates a Dataset like object that supports "." access of tensors and appends/extends to the tensors.
        This is used as sample_out in hub transforms.
        """
        self.tensors = all_tensors or {}
        self.slice_list = slice_list or []

    def __len__(self):
        return min(len(self[tensor]) for tensor in self.tensors)

    def __getattr__(self, name):
        if name not in self.tensors:
            self.tensors[name] = TransformTensor(name=name, dataset=self)
        return self.tensors[name][self.slice_list]

    def __getitem__(self, slice_):
        if isinstance(slice_, str):
            return self.__getattr__(slice_)
        assert isinstance(slice_, (slice, int))
        new_slice_list = self.slice_list + [slice_]
        return TransformDataset(all_tensors=self.tensors, slice_list=new_slice_list)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def append(self, sample, skip_ok=False):
        if not skip_ok:
            for k in self.tensors:
                if k not in sample:
                    raise TensorDoesNotExistError(k)
        if len(set(map(len, (self[k] for k in sample)))) != 1:
            raise ValueError("All tensors are expected to have the same length.")
        for k, v in sample.items():
            self[k].append(v)
