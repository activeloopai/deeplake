from typing import Iterable, List
from collections import OrderedDict


def subscript_namedtuple(T: str, fields: List[str]):
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

    class SubscriptNamedTuple(object):
        # For pytorch dataloader compatibility
        __class__ = tuple  # type: ignore
        _fields = fields

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
                T,
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

    return SubscriptNamedTuple
