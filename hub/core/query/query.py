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
        progress_callback: Callable[[int], None] = lambda *_: None,
    ):
        self._dataset = dataset
        self._query = query
        self._pg_callback = progress_callback
        self._cquery = compile(query, "", "eval")
        self._tensors = [tensor for tensor in dataset.tensors.keys() if tensor in query]
        self._np_access: List[NP_ACCESS] = [
            _get_np(dataset, block) for block in expand(dataset, self._tensors)
        ]
        self._wrappers = self._export_tensors()

    def execute(self) -> List[int]:
        print("execute()", len(self._dataset))
        idx_map: List[int] = list()
        max_size = len(self._dataset)
        num_samples_processed = 0
        for f in self._np_access:
            cache = {tensor: f(tensor) for tensor in self._tensors}
            for local_idx, idx in enumerate(f("index")):
                if idx >= max_size:
                    break

                p = {
                    tensor: self._wrap_value(tensor, cache[tensor][local_idx])
                    for tensor in self._tensors
                }
                num_samples_processed += 1
                if eval(self._cquery, p):
                    idx_map.append(int(idx))
                    self._pg_callback(int(idx), True)
                else:
                    self._pg_callback(int(idx), False)
        print("execute() Done. num_samples_processed: ", num_samples_processed)
        assert num_samples_processed == len(self._dataset)
        return idx_map

    def _wrap_value(self, tensor, val):
        if tensor in self._wrappers:
            return self._wrappers[tensor].with_value(val)
        else:
            return val

    def _export_tensors(self):
        r = {}
        r.update(
            {
                tensor_key: export_tensor(tensor)
                for tensor_key, tensor in self._dataset.tensors.items()
                if not "/" in tensor_key
            }
        )

        r.update(
            {
                tensor_key: GroupTensor(self._dataset, tensor_key.split("/")[0])
                for tensor_key in self._dataset.tensors.keys()
                if "/" in tensor_key
            }
        )

        return r


def _get_np(dataset: Dataset, block: IOBlock) -> NP_ACCESS:
    idx = block.indices()

    def f(tensor: str) -> NP_RESULT:
        if tensor == "index":
            return numpy.array(idx)
        else:
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
    def __init__(self, dataset: Dataset, prefix: str) -> None:
        super().__init__()
        self.prefix = prefix
        self.dataset = dataset
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
            if "/" in tensor:
                prefix = self.prefix + "/" + tensor.split("/")[0]
                r[tensor] = GroupTensor(self.dataset, prefix)
            else:
                t = self.dataset.tensors[f"{self.prefix}/{tensor}"]
                r[tensor] = export_tensor(t)

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
