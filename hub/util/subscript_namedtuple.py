from typing import List
from collections import OrderedDict


def create_parametrized_named_tuple(T: str, fields: List[str]):
    class DummySubClass(SubscriptNamedTuple):
        _fields = fields
        _T = T
    return DummySubClass


class SubscriptNamedTuple(object):
    """
    Similar to `collections.namedtuple`, but uses subscripting instead of
    dot operator to access fields to allow arbitrary string field names.

    Example:

    T = namedtuple("T", ["a", "b", "c"])
    t = T(1, 2, 3)
    # or
    t = T(a=1, b=2, c=3)

    assert t['a'] == 1
    assert t['b'] == 2
    assert t['c'] == 3

    for x in t:
        print(x)

    # >>> 1
    # >>> 2
    # >>> 3
    """

    # For pytorch dataloader compatibility
    __class__ = tuple  # type: ignore

    def __init__(self, *args, **kwargs):
        self._dict = OrderedDict()
        for i in range(len(args)):
            self[self._fields[i]] = args[i]
        for f in self._fields:
            if f in kwargs:
                self[f] = kwargs[f]
        for k, v in kwargs.items():
            if k not in self._fields:
                self[k] = v

    def __contains__(self, k):
        return k in self._dict

    def __setitem__(self, k, v):
        self._dict[k] = v

    def __getitem__(self, k):
        return self._dict[k]

    def __reduce__(self):
        return (_InitializeParameterized(), (self._T, self._fields), self.__dict__)

    def __getstate__(self):
        self.__class__ = SubscriptNamedTuple
        return self.__dict__

    def __setstate__(self, state):
        self.__class__ = tuple
        self.__dict__ = state

    def __iter__(self):
        for v in self._dict.values():
            yield v

    def __getattribute__(self, attr):
        try:
            return object.__getattribute__(self, attr)
        except AttributeError:
            return getattr(self._dict, attr)

    def __len__(self):
        return len(self._dict)

    def __repr__(self):
        return "%s(%s)" % (
            self._T,
            ", ".join(["%s=%s" % (k, v) for k, v in self.items()]),
        )

    def __eq__(self, other):
        try:
            if len(self) != len(other):
                return
            other_keys = list(other.keys())
            for i, (k, v) in enumerate(self.items()):
                if other_keys[i] != k:
                    return False
                if other[k] != v:
                    return False
            return True
        except Exception:
            return False


class _InitializeParameterized(object):
    """When called with the param value as the only argument, returns an un-initialized instance of the parameterized class.
    Subsequent __setstate__ will be called by pickle."""
    def __call__(self, T, fields):
        obj = _InitializeParameterized()
        obj.__class__ = create_parametrized_named_tuple(T, fields)
        return obj