from abc import ABC
from typing import Any, Dict, Union

import hub
import json
import numpy as np

from hub.core.index.index import Index
from hub.core.tensor import Tensor

enable_cache = True


class DatasetQuery:
    """Dataset query function which is defined by the query string.
    Designed not to be used directly, but passed as a function parameter to
    `dataset.filter()`

    Example:

    >>> query = DatasetQuery(dataset, 'images[1:3].min == 0')
    >>> query = DatasetQuery(dataset, 'labels > 5')
    >>> query(sample_in)
    True
    """

    def __init__(self, dataset: hub.Dataset, query: str):
        self._dataset = dataset
        self._query = query
        self.global_vars: Dict[str, Any] = dict()
        self.cache: Dict[str, np.ndarray] = dict()
        self.locals: Dict[str, Any] = self._export_tensors(dataset)
        self.index: Index = Index()

    def __call__(self, *args: Any) -> bool:
        return self._call_eval(*args)

    def _call_eval(self, sample_in: hub.Dataset):
        self.index = sample_in.index  # type: ignore

        return eval(self._query, self.global_vars, self.locals)

    def _export_tensors(self, sample_in):
        bindings: Dict[str, EvalObject] = {"dataset": EvalDataset(self)}

        for key, tensor in sample_in.tensors.items():
            if tensor.htype == "class_label":
                bindings[key] = EvalLabelClassTensor(self, tensor)
                for label in tensor.info["class_names"]:
                    bindings[label] = label
            elif tensor.htype == "text":
                bindings[key] = EvalTextTesor(self, tensor)
            elif tensor.htype == "json":
                bindings[key] = EvalJsonTensor(self, tensor)
            elif "/" in key:
                group, tensor_name = key.split("/")

                if not group in bindings:
                    bindings[group] = EvalGroupTensor(self, group)
                bindings[group].extend(tensor_name, tensor)  # type: ignore

            else:
                bindings[key] = EvalGenericTensor(self, tensor)

        return bindings

    def __repr__(self) -> str:
        return f"'{self._query}' on {self._dataset}"


class EvalObject(ABC):
    def __init__(self, query: DatasetQuery) -> None:
        self.query = query


class EvalDataset(EvalObject):
    def __init__(self, query: DatasetQuery) -> None:
        super().__init__(query)

    def __getitem__(self, item):
        return self.query._dataset[self.query.index].__getitem__(item)


class EvalTensorObject(EvalObject):
    def __init__(self, query: DatasetQuery, tensor: Tensor) -> None:
        super().__init__(query)
        self._tensor = tensor
        self._is_data_cacheable = self._tensor.chunk_engine.is_data_cachable

    def at_index(self, index):
        self.query.index = index

    def is_scalar(self):
        return self.numpy().size == 1  # type: ignore

    def as_scalar(self):
        if self.is_scalar():
            return self.numpy()[0]
        else:
            raise ValueError("Not a scalar")

    # FIXME bad code here
    def _get_cached_numpy(self):
        cache = self.query.cache
        if self._tensor.key not in cache:
            full_tensor = Tensor(
                self._tensor.key,
                self._tensor.storage,
                self._tensor.version_state,
                Index(),
                False,
                chunk_engine=self._tensor.chunk_engine,
            )
            cache[self._tensor.key] = full_tensor.numpy()

        return cache[self._tensor.key]

    def numpy(self):
        """Retrives np.ndarray or scalar value"""
        if self._is_data_cacheable and enable_cache:
            cache = self._get_cached_numpy()
            idx = self.query.index.values[0].value
            return cache[idx]
        else:
            return self._tensor[self.query.index].numpy()


class ScalarTensorObject(EvalObject):
    """Abstract subclass defining operations on a scalars.
    Requires override of  `as_scalar()` to enable operations set of [ = > < >= <= != ]
    """

    def as_scalar(self) -> Any:
        raise NotImplementedError

    def __eq__(self, o: object) -> bool:
        return self.as_scalar() == o

    def __lt__(self, o: object) -> bool:
        return self.as_scalar() < o

    def __le__(self, o: object) -> bool:
        return self.as_scalar() <= o

    def __gt__(self, o: object) -> bool:
        return self.as_scalar() > o

    def __ge__(self, o: object) -> bool:
        return self.as_scalar() >= o

    def __ne__(self, o: object) -> bool:
        return self.as_scalar() != o


class EvalGenericTensor(EvalTensorObject, ScalarTensorObject):
    """Wrapper of tensor for evaluation function `eval()` which provides simplified interface to work
    with a tensor.

    Defines functions available on tensor `t`
    >>> t = tensor(np.array(...))
    >>> t.min == t.numpy.min()
    >>> t.max == t.numpy.max()
    >>> t.mean == t.numpy.mean()
    >>> t.shape == t.numpy.shape
    >>> t.size == t.numpy.size


    Enables scalar operations, if tensor is size of 1.
    >>> t = tensor([4])
    >>> t > 3 or t < 5
    True
    """

    def __init__(self, query: DatasetQuery, tensor: Tensor) -> None:
        super().__init__(query, tensor)

    @property
    def min(self):
        """Returns numpy.min() for the tensor"""
        return np.amin(self.numpy())

    @property
    def max(self):
        """Returns numpy.max() for the tensor"""
        return np.amax(self.numpy())

    @property
    def mean(self):
        """Returns numpy.mean() for the tensor"""
        return self.numpy().mean()  # type: ignore

    @property
    def shape(self):
        """Returns shape of the underlying numpy array"""
        return self.numpy().shape  # type: ignore

    @property
    def size(self):
        """Returns size of the underlying numpy array"""
        return self.numpy().size  # type: ignore

    def contains(self, o: object) -> bool:
        return bool(np.any(self.numpy() == o))

    def __getitem__(self, item):
        """Returns subscript of underlying numpy array or a scalar"""
        val = self.numpy()[item]

        if isinstance(val, np.ndarray):
            return EvalNumpyObject(val)
        else:
            return val


class EvalNumpyObject(ScalarTensorObject):
    def __init__(self, arr: np.ndarray) -> None:
        self.arr = arr

    def as_scalar(self) -> Any:
        return self.arr[0]

    @property
    def min(self):
        """Returns numpy.min() for the wrapped np.ndarray"""
        return np.amin(self.arr)

    @property
    def max(self):
        """Returns numpy.max() for the wrapped np.ndarray"""
        return np.amax(self.arr)

    @property
    def mean(self):
        """Returns numpy.mean() for the wrapped np.ndarray"""
        return np.mean(self.arr)

    @property
    def shape(self):
        """Returns shape of the underlying numpy array"""
        return self.arr.shape  # type: ignore

    @property
    def size(self):
        """Returns size of the underlying numpy array"""
        return self.arr.size  # type: ignore

    def contains(self, o: object) -> bool:
        return bool(np.any(self.arr == o))

    def __getitem__(self, item):
        """Returns subscript of underlying numpy array or a scalar"""
        val = self.arr[item]

        if isinstance(val, np.ndarray):
            if val.size == 1:
                return val[0]
            else:
                return EvalNumpyObject(val)
        else:
            return val


class EvalLabelClassTensor(EvalTensorObject, ScalarTensorObject):
    """Wrapper for `tensor(htype = 'label_class')`. Provides equality operation
    for labeled data.

    Example:
        >>> query('labels == "dog"')
    """

    def __init__(self, query: DatasetQuery, tensor: Tensor) -> None:
        super().__init__(query, tensor)

        # FIXME tensor.info should be resolvable class
        self._class_names = tensor.info["class_names"]  # type: ignore

    def __eq__(self, o: object) -> bool:
        if not self.is_scalar():
            raise ValueError("Vector and scalars are not comparable")

        if isinstance(o, str):
            return self._class_names[self.as_scalar()] == o
        else:
            return super().__eq__(o)

    def contains(self, o: object) -> bool:
        v = self.numpy() == o
        if isinstance(v, np.ndarray):
            return v.any()  # type: ignore
        else:
            return v

    def __iter__(self):
        return iter(self.numpy())

    def __getitem__(self, item):
        if not self.is_scalar():
            return EvalLabelClassTensor(self.query, self._tensor[item])
        else:
            raise ValueError("Scalar not subscriptable")


class EvalTextTesor(EvalTensorObject, ScalarTensorObject):
    """Wrapper for `tensor(htype = 'text'). Provides equality operation
    for text data.

    Example:
        >>> query('text == "some_string"')
        >>> query('len(text) == 0')
    """

    def __init__(self, query: DatasetQuery, tensor) -> None:
        super().__init__(query, tensor)

    def contains(self, o: object) -> bool:
        return bool(np.any(self.numpy() == o))

    def as_scalar(self):
        assert self.is_scalar()
        return self.numpy()[0]

    def __len__(self):
        return len(self.as_scalar()) if self.is_scalar() else len(self.numpy())


class EvalJsonTensor(EvalTensorObject):
    """Wrapper for `tensor(htype = 'json'). Expose json dictionary to user

    Example:
        >>> query('json["attribute"]["nested_attribute"] == "some_value"')
    """

    def __init__(self, query: DatasetQuery, tensor) -> None:
        super().__init__(query, tensor)

    def as_scalar(self):
        assert self.is_scalar()
        return json.loads(self.numpy()[0])

    def __getitem__(self, item):
        return self.as_scalar()[item]


class EvalGroupTensor(EvalObject):
    """Wrapper around tensor groups.

    Example:
        >>> query('images.image1.mean == images.image2.mean')
    """

    def __init__(self, query: DatasetQuery, group) -> None:
        super().__init__(query)
        self.group = group
        self._map: Dict[str, EvalObject] = dict()

    def extend(self, name: str, tensor: Tensor):
        if tensor.htype == "class_label":
            self._map[name] = EvalLabelClassTensor(self.query, tensor)
        elif tensor.htype == "text":
            self._map[name] = EvalTextTesor(self.query, tensor)
        elif tensor.htype == "json":
            self._map[name] = EvalJsonTensor(self.query, tensor)
        else:
            self._map[name] = EvalGenericTensor(self.query, tensor)

    def __getattr__(self, name):
        return self._map[name]
