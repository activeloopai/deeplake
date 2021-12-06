from abc import ABC
from typing import Any, Dict

import hub
import json
import numpy as np

from hub.core.tensor import Tensor


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

    def __call__(self, *args: Any) -> bool:
        return self._call_eval(*args)

    def _call_eval(self, sample_in: hub.Dataset):
        return eval(
            self._query, {}, self._export_tensors(sample_in)
        )  # TODO export tensors create new bindings. Can we update instead ?

    def _export_tensors(self, sample_in):
        bindings = {"dataset": sample_in}

        for key, tensor in sample_in.tensors.items():
            if tensor.htype == "class_label":
                bindings[key] = EvalLabelClassTensor(tensor)
                for label in tensor.info["class_names"]:
                    bindings[label] = label
            elif tensor.htype == "text":
                bindings[key] = EvalTextTesor(tensor)
            elif tensor.htype == "json":
                bindings[key] = EvalJsonTensor(tensor)
            elif "/" in key:
                group, tensor_name = key.split("/")

                if not group in bindings:
                    bindings[group] = EvalGroupTensor(group)
                bindings[group].extend(tensor_name, tensor)

            else:
                bindings[key] = EvalGenericTensor(tensor)

        return bindings

    def __repr__(self) -> str:
        return f"'{self._query}' on {self._dataset}"


class EvalObject(ABC):
    pass


class EvalTensorObject(EvalObject):
    def __init__(self, tensor: Tensor) -> None:
        super().__init__()
        self._tensor = tensor
        self._cached_value: Any = None
        self._is_cached_value: bool = False

    def is_scalar(self):
        return self.numpy().size == 1  # type: ignore

    def as_scalar(self):
        if self.is_scalar():
            return self.numpy()[0]
        else:
            raise ValueError("Not a scalar")

    def numpy(self):
        """Retrives and caches value"""
        if not self._is_cached_value:
            self._cached_value = self._tensor.numpy()
            self._is_cached_value = True

        return self._cached_value


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

    def __init__(self, tensor: Tensor) -> None:
        super().__init__(tensor)

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
        return np.mean(self.numpy())

    @property
    def shape(self):
        """Returns shape of the underlying numpy array"""
        return self.numpy().shape  # type: ignore

    @property
    def size(self):
        """Returns size of the underlying numpy array"""
        return self.numpy().size  # type: ignore

    def contains(self, o: object) -> bool:
        return any(self.numpy() == o)  # type: ignore

    def __getitem__(self, item):
        """Returns subscript of underlying numpy array or a scalar"""
        val = self.numpy()[item]

        if isinstance(val, np.ndarray):
            return EvalGenericTensor(self._tensor[item])
        else:
            return val


class EvalLabelClassTensor(EvalTensorObject, ScalarTensorObject):
    """Wrapper for `tensor(htype = 'label_class')`. Provides equality operation
    for labeled data.

    Example:
        >>> query('labels == "dog"')
    """

    def __init__(self, tensor: Tensor) -> None:
        super().__init__(tensor)

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
        return any(self.numpy() == o)  # type: ignore


class EvalTextTesor(EvalTensorObject, ScalarTensorObject):
    """Wrapper for `tensor(htype = 'text'). Provides equality operation
    for text data.

    Example:
        >>> query('text == "some_string"')
        >>> query('len(text) == 0')
    """

    def __init__(self, tensor) -> None:
        super().__init__(tensor)

    def contains(self, o: object) -> bool:
        return any(self.numpy() == o)  # type: ignore

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

    def __init__(self, tensor) -> None:
        super().__init__(tensor)

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

    def __init__(self, group) -> None:
        super().__init__()
        self.group = group
        self._map: Dict[str, EvalObject] = dict()

    def extend(self, name: str, tensor: Tensor):
        if tensor.htype == "class_label":
            self._map[name] = EvalLabelClassTensor(tensor)
        elif tensor.htype == "text":
            self._map[name] = EvalTextTesor(tensor)
        elif tensor.htype == "json":
            self._map[name] = EvalJsonTensor(tensor)
        else:
            self._map[name] = EvalGenericTensor(tensor)

    def __getattr__(self, name):
        return self._map[name]
