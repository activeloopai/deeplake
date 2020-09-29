import numpy as np

from .collections import dataset
from .collections.dataset.core import Transform
from .collections import tensor
from .collections.dataset import load
from .collections.client_manager import init
import hub.config
import hub.api.dataset


def local_mode():
    hub.config.HUB_REST_ENDPOINT = hub.config.HUB_LOCAL_REST_ENDPOINT


def dtype(*args, **kwargs):
    return np.dtype(*args, **kwargs)


def open(url: str = None, mode: str = None, shape=None, dtype=None, token=None):
    assert url is not None
    shape = shape or (None,)

    assert len(shape) == 1

    return hub.api.dataset.Dataset(
        url, token, num_samples=shape[0], mode=mode, dtype=dtype
    )
