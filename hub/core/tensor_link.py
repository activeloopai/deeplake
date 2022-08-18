from typing import Callable
from hub.core.index import Index
from hub.constants import _NO_LINK_UPDATE
import inspect
import hub
import inspect
from hub.util.generate_id import generate_id
import numpy as np


class _TensorLinkTransform:
    def __init__(self, f):
        self.name = f.__name__
        self.f = f
        spec = inspect.getfullargspec(f)
        self.multi_arg = len(spec.args) > 1 or spec.varargs or spec.varkw
        self.kwargs = [k for k in ("index", "sub_index", "partial") if k in spec.args]

    def __call__(self, *args, **kwargs):
        if self.multi_arg:
            out_kwargs = {k: v for k, v in kwargs.items() if k in self.kwargs}
            return self.f(*args, **out_kwargs)
        return self.f(args[0])

    def __str__(self):
        return f"TensorLinkTransform[{self.name}]"


link = _TensorLinkTransform


@link
def append_id(sample, link_creds=None):
    return generate_id(np.uint64)


@link
def append_test(sample, link_creds=None):
    return sample


@link
def update_test(
    new_sample, old_value, sub_index: Index, partial: bool, link_creds=None
):
    return old_value


@link
def append_info(sample, link_creds=None):
    meta = {}
    if isinstance(sample, hub.core.linked_sample.LinkedSample):
        sample = read_linked_sample(
            sample.path, sample.creds_key, link_creds, verify=False
        )
    if isinstance(sample, hub.core.sample.Sample):
        meta = sample.meta
        meta["modified"] = False
    return meta


@link
def update_info(
    new_sample, old_value, sub_index: Index, partial: bool, link_creds=None
):
    if partial:
        meta = old_value.data()
        if "modified" in meta:
            meta["modified"] = True
            return meta
    else:
        return append_info.f(new_sample, link_creds)
    return _NO_LINK_UPDATE


@link
def append_shape(sample, link_creds=None):
    if isinstance(sample, hub.core.linked_sample.LinkedSample):
        sample = read_linked_sample(
            sample.path, sample.creds_key, link_creds, verify=False
        )
    return np.array(
        getattr(sample, "shape", None) or np.array(sample).shape, dtype=np.int64
    )


@link
def append_len(sample, link_creds=None):
    return 0 if sample is None else len(sample)


_funcs = {k: v for k, v in globals().items() if isinstance(v, link)}


def _register_link_transform(fname: str, func: Callable):
    _funcs[fname] = _TensorLinkTransform(func)


def _unregister_link_transform(fname: str):
    _funcs.pop(fname, None)


def get_link_transform(fname: str):
    return _funcs[fname]


def read_linked_sample(
    sample_path: str, sample_creds_key: str, link_creds, verify: bool
):
    if sample_path.startswith(("gcs://", "gcp://", "gs://", "s3://")):
        provider_type = "s3" if sample_path.startswith("s3://") else "gcs"
        storage = link_creds.get_storage_provider(sample_creds_key, provider_type)
        return hub.read(sample_path, storage=storage, verify=verify)
    elif sample_path.startswith(("http://", "https://")):
        creds = link_creds.get_creds(sample_creds_key)
        return hub.read(sample_path, verify=verify, creds=creds)
    return hub.read(sample_path, verify=verify)
