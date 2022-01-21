from typing import Any, Callable, List, Union
from hub.core.dataset import Dataset
from hub.core.io import IOBlock, SampleStreaming
from hub.core.index import Index
from hub.core.tensor import Tensor


import numpy

NP_RESULT = Union[numpy.ndarray, List[numpy.ndarray]]
NP_ACCESS = Callable[[str], NP_RESULT]


class DatasetQuery:
    def __init__(
        self,
        dataset,
        query: str,
        progress_callback: Callable[[int], None] = lambda _: None,
    ):
        self._dataset = dataset
        self._query = query
        self._pg_callback = progress_callback
        self._cquery = compile(query, "", "eval")
        self._tensors = [
            tensor
            for tensor in dataset.tensors.keys()
            if normalize_query_tensors(tensor) in query
        ]
        self._np_access: List[NP_ACCESS] = [
            _get_np(dataset, block) for block in expand(dataset, self._tensors)
        ]
        self._wrappers = self._export_tensors()
        self._groups = self._export_groups(self._wrappers)

    def execute(self) -> List[int]:
        idx_map: List[int] = list()
        max_size = len(self._dataset)

        for f in self._np_access:
            cache = {tensor: f(tensor) for tensor in self._tensors}
            for local_idx in range(max_size):
                p = {
                    tensor: self._wrap_value(tensor, cache[tensor][local_idx])
                    for tensor in self._tensors
                }

                p.update(self._groups)

                if eval(self._cquery, p):
                    idx_map.append(local_idx)
                self._pg_callback(1)

        return idx_map

    def _wrap_value(self, tensor, val):
        if tensor in self._wrappers:
            return self._wrappers[tensor].with_value(val)
        else:
            return val

    def _export_tensors(self):
        return {
            tensor_key: export_tensor(tensor)
            for tensor_key, tensor in self._dataset.tensors.items()
        }

    def _export_groups(self, wrappers):
        return {
            extract_prefix(tensor_key): GroupTensor(
                self._dataset, wrappers, extract_prefix(tensor_key)
            )
            for tensor_key in self._dataset.tensors.keys()
            if "/" in tensor_key
        }


def normalize_query_tensors(tensor_key: str) -> str:
    return tensor_key.replace("/", ".")


def extract_prefix(tensor_key: str) -> str:
    return tensor_key.split("/")[0]


def _get_np(dataset: Dataset, block: IOBlock) -> NP_ACCESS:
    idx = block.indices()

    def f(tensor: str) -> NP_RESULT:
        tensor_obj = dataset.tensors[tensor]
        tensor_obj.index = Index()
        return tensor_obj[idx].numpy(aslist=tensor_obj.is_dynamic)

    return f


def expand(dataset, tensor: List[str]) -> List[IOBlock]:
    return SampleStreaming(dataset, tensor).list_blocks()


def export_tensor(tensor: Tensor):
    if tensor.htype == "class_label":
        return ClassLabelsTensor(tensor)

    return EvalObject()


class EvalObject:
    def __init__(self) -> None:
        super().__init__()
        self._val: Any = None

    @property
    def val(self):
        return self._val

    def with_value(self, v: Any):
        self._val = v
        return self

    def contains(self, v: Any):
        return v in self.val

    def __getitem__(self, item):
        r = EvalObject()
        return r.with_value(self._val[item])

    @property
    def min(self):
        """Returns numpy.min() for the tensor"""
        return numpy.amin(self.val)

    @property
    def max(self):
        """Returns numpy.max() for the tensor"""
        return numpy.amax(self.val)

    @property
    def mean(self):
        """Returns numpy.mean() for the tensor"""
        return self.val.mean()

    @property
    def shape(self):
        """Returns shape of the underlying numpy array"""
        return self.val.shape  # type: ignore

    @property
    def size(self):
        """Returns size of the underlying numpy array"""
        return self.val.size  # type: ignore

    def __eq__(self, o: object) -> bool:
        return self.val == o

    def __lt__(self, o: object) -> bool:
        return self.val < o

    def __le__(self, o: object) -> bool:
        return self.val <= o

    def __gt__(self, o: object) -> bool:
        return self.val > o

    def __ge__(self, o: object) -> bool:
        return self.val >= o

    def __ne__(self, o: object) -> bool:
        return self.val != o


class GroupTensor:
    def __init__(self, dataset: Dataset, wrappers, prefix: str) -> None:
        super().__init__()
        self.prefix = prefix
        self.dataset = dataset
        self.wrappers = wrappers
        self._subgroup = self.expand()

    def __getattr__(self, __name: str) -> Any:
        return self._subgroup[self.normalize_key(__name)]

    def expand(self):
        r = {}
        for tensor in [
            self.normalize_key(t)
            for t in self.dataset.tensors
            if t.startswith(self.prefix)
        ]:
            prefix = self.prefix + "/" + extract_prefix(tensor)
            if "/" in tensor:
                r[tensor] = GroupTensor(self.dataset, self.wrappers, prefix)
            else:
                r[tensor] = self.wrappers[prefix]

        return r

    def normalize_key(self, key: str) -> str:
        return key.replace(self.prefix + "/", "")


class ClassLabelsTensor(EvalObject):
    def __init__(self, tensor: Tensor) -> None:
        super().__init__()
        self._tensor = tensor

        _classes = tensor.info["class_names"]  # type: ignore
        self._classes_dict = {v: idx for idx, v in enumerate(_classes)}

    def __eq__(self, o: object) -> bool:
        if isinstance(o, str):
            return self.val == self._classes_dict[o]
        else:
            return self.val == o

    def __lt__(self, o: object) -> bool:
        if isinstance(o, str):
            raise ValueError("label class is not comparable")
        return self.val < o

    def __le__(self, o: object) -> bool:
        if isinstance(o, str):
            raise ValueError("label class is not comparable")
        return self.val <= o

    def __gt__(self, o: object) -> bool:
        if isinstance(o, str):
            raise ValueError("label class is not comparable")
        return self.val > o

    def __ge__(self, o: object) -> bool:
        if isinstance(o, str):
            raise ValueError("label class is not comparable")
        return self.val >= o

    def __ne__(self, o: object) -> bool:
        if isinstance(o, str):
            return self.val != self._classes_dict[o]
        else:
            return self.val != o
