import numpy as np

from .collections import dataset
from .collections.dataset.core import Transform
from .collections import tensor
from .collections.dataset import load
from .collections.client_manager import init
import hub.config
import hub.api.dataset
from hub.compute.pipeline import transform


def local_mode():
    hub.config.HUB_REST_ENDPOINT = hub.config.HUB_LOCAL_REST_ENDPOINT


def dev_mode():
    hub.config.HUB_DEV_REST_ENDPOINT = hub.config.HUB_DEV_REST_ENDPOINT


def dtype(*args, **kwargs):
    return np.dtype(*args, **kwargs)


def open(url: str = None, mode: str = None, shape=None, dtype=None, token=None):
    """Open a new or existing dataset for read/write

    Parameters
    ----------
    url: str
        The url where dataset is located/should be created
    mode: str
        Python way to tell whether dataset is for read or write (ex. "r", "w", "w+")
    shape: tuple, optional
        Tuple with (num_samples,) format, where num_samples is number of samples
    dtype: optional
        Describes the data of a single sample. Hub features are used for that
    token: str or dict, optional
        If url is refering to a place where authorization is required,
        token is the parameter to pass the credentials, it can be filepath or dict
    """
    assert url is not None
    shape = shape or (None,)

    assert len(shape) == 1

    return hub.api.dataset.Dataset(
        url, token=token, shape=shape, mode=mode, dtype=dtype
    )


dev_mode()