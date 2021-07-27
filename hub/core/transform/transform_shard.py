from hub.core.transform.transform_tensor import TransformDatasetTensor
from hub.util.exceptions import TensorDoesNotExistError

# TODO: add tests


def slice_with_list(item, slice_list):
    """Slice an object with a list of slices."""
    sliced = item
    for slice_ in slice_list:
        sliced = sliced[slice_]
    return sliced


class TransformDatasetShard:
    def __init__(self, all_tensors=None, slice_list=None):
        self.tensors = all_tensors or {}
        self.slice_list = slice_list or []

    def __len__(self):
        return min(len(self[tensor]) for tensor in self.tensors)

    def _check_length_equal(self):
        lengths = [len(self[tensor]) for tensor in self.tensors]
        if any(l != lengths[0] for l in lengths):
            raise Exception  # TODO proper exception

    def __getattr__(self, name):
        if name not in self.tensors:
            self.tensors[name] = TransformDatasetTensor()
        return self.tensors[name][self.slice_list]

    def __getitem__(self, slice_):
        if isinstance(slice_, str):
            return self.__getattr__(slice_)
        assert isinstance(slice_, (slice, int))
        new_slice_list = self.slice_list + [slice_]
        return TransformDatasetShard(
            all_tensors=self.tensors, slice_list=new_slice_list
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
