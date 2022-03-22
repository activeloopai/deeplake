from typing import Callable
from hub.core.index import Index
from hub.constants import _NO_LINK_UPDATE
import inspect
import hub


class _TensorLinkTransform:
    def __init__(self, f):
        self.name = f.__name__
        self.f = f
        spec = inspect.getfullargspec(f)
        self.multi_arg = len(spec.args) > 1 or spec.varargs or spec.varkw
        self.kwargs = [k for k in ("index", "partial") if k in spec.args]

    def __call__(self, *args, **kwargs):
        if self.multi_arg:
            out_kwargs = {k: v for k, v in kwargs.items() if k in self.kwargs}
            return self.f(*args, **out_kwargs)
        return self.f(args[0])

    def __str__(self):
        return f"TensorLinkTransform[{self.name}]"


link = _TensorLinkTransform


@link
def append_test(sample):
    return sample


@link
def update_test(new_sample, old_value, sub_index: Index, partial: bool):
    return old_value


@link
def append_info(sample):
    if isinstance(sample, hub.core.sample.Sample):
        meta = sample.meta
        meta["modified"] = False
        return meta
    return {}


@link
def update_info(new_sample, old_value, sub_index: Index, partial: bool):
    if isinstance(new_sample, hub.core.sample.Sample) and not partial:
        meta = old_value.data()
        if "modified" in meta:
            meta["modified"] = True
            return meta
    return _NO_LINK_UPDATE


_funcs = {k: v for k, v in globals().items() if isinstance(v, link)}


def _register_link_transform(fname: str, func: Callable):
    _funcs[fname] = _TensorLinkTransform(func)


def _unregister_link_transform(fname: str):
    _funcs.pop(fname, None)


def get_link_transform(fname: str):
    return _funcs[fname]


class LinkTransformTestContext:
    def __init__(self, func: Callable, name: str):
        self.func = func
        self.name = name

    def __enter__(self):
        _register_link_transform(self.name, self.func)

    def __exit__(self, *args, **kwargs):
        _unregister_link_transform(self.name)
