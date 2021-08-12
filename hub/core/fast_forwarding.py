from hub.constants import ENCODING_DTYPE
import numpy as np
from hub.core.meta.encode.shape import ShapeEncoder
import hub


def _check_version(v):
    """Raises exceptions for incompatible versions. Returns True if no fast forwarding is required (False if otherwise)."""

    # TODO: if `v` is newer than the current version, raise an exception

    if v == hub.__version__:
        return True

    return False

def ffw(func):
    # TODO: docstring

    def decor(inp, **kwargs):
        v = inp.version
        if not _check_version(v):
            out = func(inp, v, **kwargs)
            inp.version = hub.__version__
            return out

    return decor


@ffw
def ffw_tensor_meta(tensor_meta, version):
    if version in ("2.0.2", "2.0.3", "2.0.4", "2.0.5"):
        if len(tensor_meta.min_shape) == 0:
            tensor_meta.min_shape = [1]
            tensor_meta.max_shape = [1]


@ffw
def ffw_chunk_id_encoder(chunk_id_encoder, version):
    pass


@ffw
def ffw_dataset_meta(dataset_meta, version):
    pass


@ffw
def ffw_chunk(chunk, version):
    if version in ("2.0.2", "2.0.3", "2.0.4", "2.0.5"):
        shapes = chunk.shapes_encoder
        if shapes.dimensionality == 0:
            a = shapes.array

            if len(a) > 1:
                raise ValueError()  # TODO: exceptions.py

            shapes._encoded = np.array([[1, a[0][0]]], dtype=ENCODING_DTYPE)