"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import time
from collections import abc
from math import gcd

from numpy.lib.arraysetops import isin

from hub_v1 import defaults
from hub_v1.exceptions import ShapeLengthException


def _flatten(list_):
    """
    Helper function to flatten the list
    """
    return [item for sublist in list_ for item in sublist]


def gcp_creds_exist():
    """Checks if credentials exists"""

    try:
        import os

        env = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if env is not None:
            return True
        from google.cloud import storage

        storage.Client()
    except Exception:
        return False
    return True


def s3_creds_exist():
    import boto3

    sts = boto3.client("sts")
    try:
        sts.get_caller_identity()
    except Exception:
        return False
    return True


def azure_creds_exist():
    """Checks if credentials exists"""

    import os

    env = os.getenv("ACCOUNT_KEY")
    return env is not None


def hub_creds_exist():
    """Checks if credentials exists"""

    import os

    env = os.getenv("ACTIVELOOP_HUB_PASSWORD")
    return env is not None


def minio_creds_exist():
    """Checks if credentials exists"""

    import os

    env1 = os.getenv("ACTIVELOOP_MINIO_KEY")
    env2 = os.getenv("ACTIVELOOP_MINIO_SECRET_ACCESS_KEY")
    return env1 is not None and env2 is not None


def pandas_loaded():
    try:
        import pandas as pd

        pd.__version__
    except ImportError:
        return False
    return True


def pytorch_loaded():
    try:
        import torch

        torch.__version__
    except ImportError:
        return False
    return True


def ray_loaded():
    try:
        import ray

        ray.__version__
    except ImportError:
        return False
    return True


def dask_loaded():
    try:
        import dask

        dask.__version__
    except ImportError:
        return False
    return True


def tensorflow_loaded():
    try:
        import tensorflow

        tensorflow.__version__
    except ImportError:
        return False
    return True


def tfds_loaded():
    try:
        import tensorflow_datasets

        tensorflow_datasets.__version__
    except ImportError:
        return False
    return True


def transformers_loaded():
    try:
        import transformers

        transformers.__version__
    except ImportError:
        return False
    return True


def pathos_loaded():
    try:
        import pathos

        pathos.__version__
    except ImportError:
        return False
    return True


def compute_lcm(a):
    """
    Lowest Common Multiple of a list a
    """
    if not a:
        return None
    lcm = a[0]
    for i in a[1:]:
        lcm = lcm * i // gcd(lcm, i)
    return int(lcm)


def batchify(iterable, n=1, initial=None):
    """
    Batchify an iteratable
    """
    ls = len(iterable)
    batches = []
    initial = initial or n
    batches.append(iterable[0 : min(initial, ls)])
    for ndx in range(initial, ls, n):
        batches.append(iterable[ndx : min(ndx + n, ls)])
    return batches


def _tuple_product(tuple_):
    res = 1
    for t in tuple_:
        res *= t
    return res


class Timer:
    def __init__(self, text):
        self._text = text

    def __enter__(self):
        self._start = time.time()

    def __exit__(self, *args):
        print(f"{self._text}: {time.time() - self._start}s")


def norm_shape(shape):
    shape = shape or (None,)
    if isinstance(shape, int):
        shape = (shape,)
    if not isinstance(shape, abc.Iterable):
        raise TypeError(
            f"shape is not None, int or Iterable, type(shape): {type(shape)}"
        )
    shape = tuple(shape)
    if not all([isinstance(s, int) or s is None for s in shape]):
        raise TypeError(f"Shape elements can be either int or None | shape: {shape}")
    return shape


def norm_cache(cache):
    cache = cache or 0
    if not isinstance(cache, int):
        raise TypeError("Cache should be None or int")
    return cache
