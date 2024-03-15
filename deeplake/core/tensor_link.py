from typing import Callable
import warnings
from deeplake.core.compression import to_image
from deeplake.core.index import Index
from deeplake.constants import _NO_LINK_UPDATE
import inspect
import deeplake
import inspect
import numpy as np
from os import urandom
from PIL import Image  # type: ignore
from deeplake.core.linked_sample import read_linked_sample
from deeplake.compression import IMAGE_COMPRESSION, get_compression_type
import tqdm  # type: ignore

optional_kwargs = {
    "old_value",
    "index",
    "sub_index",
    "partial",
    "factor",
    "compression",
    "htype",
    "link_creds",
    "progressbar",
    "tensor_meta",
}


class _TensorLinkTransform:
    def __init__(self, f):
        self.name = f.__name__
        self.f = f
        spec = inspect.getfullargspec(f)
        self.multi_arg = len(spec.args) > 1 or spec.varargs or spec.varkw
        self.kwargs = [k for k in optional_kwargs if k in spec.args]

    def __call__(self, *args, **kwargs):
        if self.multi_arg:
            out_kwargs = {k: v for k, v in kwargs.items() if k in self.kwargs}
            return self.f(*args, **out_kwargs)
        return self.f(args[0])

    def __str__(self):
        return f"TensorLinkTransform[{self.name}]"


link = _TensorLinkTransform


@link
def extend_id(samples, link_creds=None):
    return np.frombuffer(urandom(8 * len(samples)), dtype=np.uint64).reshape(-1)


@link
def extend_test(samples, link_creds=None):
    return samples


@link
def update_test(
    new_sample, old_value, sub_index: Index, partial: bool, link_creds=None
):
    return old_value


@link
def extend_info(samples, link_creds=None, progressbar=False):
    if progressbar:
        samples = tqdm.tqdm(samples, desc="Uploading sample meta info...")
    metas = []
    for sample in samples:
        meta = {}
        copy = True
        if isinstance(sample, deeplake.core.linked_sample.LinkedSample):
            sample = read_linked_sample(
                sample.path, sample.creds_key, link_creds, verify=False
            )
            copy = False
        if isinstance(sample, deeplake.core.sample.Sample):
            if copy:
                sample = sample.copy()
            meta = sample.meta
            meta["modified"] = False
        if isinstance(sample, deeplake.Tensor):
            meta = sample.sample_info
        metas.append(meta)
    return metas


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
        return extend_info.f([new_sample], link_creds)[0]
    return _NO_LINK_UPDATE


@link
def update_shape(new_sample, link_creds=None, tensor_meta=None):
    if new_sample is None:
        return np.zeros(1, dtype=np.int64)
    if isinstance(new_sample, deeplake.core.linked_sample.LinkedSample):
        new_sample = read_linked_sample(
            new_sample.path, new_sample.creds_key, link_creds, verify=False
        )
    if np.isscalar(new_sample) or (
        isinstance(new_sample, np.ndarray) and new_sample.shape == ()
    ):
        ret = np.array([1], dtype=np.int64)
    else:
        ret = np.array(
            getattr(new_sample, "shape", None) or np.array(new_sample).shape,
            dtype=np.int64,
        )

    if tensor_meta:
        # if grayscale being appended but tensor has rgb samples, convert shape from (h, w) to (h, w, 1)
        if (
            tensor_meta.min_shape
            and (
                tensor_meta.htype == "image"
                or (
                    IMAGE_COMPRESSION
                    in map(
                        get_compression_type,
                        (tensor_meta.sample_compression, tensor_meta.chunk_compression),
                    )
                )
            )
            and ret.shape == (2,)
            and len(tensor_meta.min_shape) == 3
        ):
            ret = np.concatenate([ret, (1,)])

        if tensor_meta.is_link and ret.size and np.prod(ret):
            tensor_meta.update_shape_interval(ret.tolist())

    return ret


@link
def extend_shape(samples, link_creds=None, tensor_meta=None):
    if isinstance(samples, np.ndarray):
        if samples.dtype != object:
            samples_shape = samples.shape
            if samples.ndim == 1:
                samples_shape = samples_shape + (1,)
            return np.tile(np.array([samples_shape[1:]]), (samples_shape[0], 1))
    if samples is None:
        return np.array([], dtype=np.int64)
    shapes = [
        update_shape.f(sample, link_creds=link_creds, tensor_meta=tensor_meta)
        for sample in samples
    ]

    max_ndim = max(map(len, shapes), default=0)
    min_ndim = min(map(len, shapes), default=0)
    mixed_ndim = max_ndim != min_ndim

    if mixed_ndim:
        for i, s in enumerate(shapes):
            if len(s) < max_ndim:
                shapes[i] = np.concatenate(
                    [s, (int(bool(np.any(s) and np.prod(s))),) * (max_ndim - len(s))]
                )
    arr = np.array(shapes)
    return arr


@link
def extend_len(samples, link_creds=None):
    return [0 if sample is None else len(sample) for sample in samples]


@link
def update_len(new_sample, link_creds=None):
    return 0 if new_sample is None else len(new_sample)


def convert_sample_for_downsampling(sample, link_creds=None):
    try:
        if isinstance(sample, deeplake.core.linked_sample.LinkedSample):
            sample = read_linked_sample(
                sample.path, sample.creds_key, link_creds, verify=False
            )
        if isinstance(sample, deeplake.core.sample.Sample):
            sample = sample.pil
        if (
            isinstance(sample, np.ndarray)
            and sample.dtype != bool
            and 0 not in sample.shape
        ):
            sample = to_image(sample)
        return sample
    except Exception as e:
        warnings.warn(f"Failed to downsample sample of type {type(sample)}")
        return None


@link
def extend_downsample(samples, factor, compression, htype, link_creds=None):
    samples = [
        convert_sample_for_downsampling(sample, link_creds) for sample in samples
    ]
    return [
        deeplake.util.downsample.downsample_sample(
            sample, factor, compression, htype, False, link_creds
        )
        for sample in samples
    ]


@link
def update_downsample(
    new_sample,
    factor,
    compression,
    htype,
    link_creds=None,
    sub_index=None,
    partial=False,
):
    new_sample = convert_sample_for_downsampling(new_sample, link_creds)
    if partial:
        for index_entry in sub_index.values:
            if not isinstance(index_entry.value, slice):
                return _NO_LINK_UPDATE
    downsampled = deeplake.util.downsample.downsample_sample(
        new_sample, factor, compression, htype, partial, link_creds
    )
    if partial:
        downsampled_sub_index = sub_index.downsample(factor, downsampled.shape)
        return downsampled_sub_index, downsampled
    return downsampled


_funcs = {k: v for k, v in globals().items() if isinstance(v, link)}


def _register_link_transform(fname: str, func: Callable):
    _funcs[fname] = _TensorLinkTransform(func)


def _unregister_link_transform(fname: str):
    _funcs.pop(fname, None)


def get_link_transform(fname: str):
    return _funcs[fname]


def cast_to_type(val, dtype):
    if isinstance(val, np.ndarray) and dtype and val.dtype != dtype:
        return val.astype(dtype)
    return val
