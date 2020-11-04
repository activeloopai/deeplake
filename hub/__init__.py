import numpy as np

from .collections import dataset
from .collections.dataset.core import Transform
from .collections import tensor
from .collections.dataset import load
from .collections.client_manager import init
import hub.config
from hub.api.dataset import Dataset
from hub.compute.pipeline import transform


def local_mode():
    hub.config.HUB_REST_ENDPOINT = hub.config.HUB_LOCAL_REST_ENDPOINT


def dev_mode():
    hub.config.HUB_DEV_REST_ENDPOINT = hub.config.HUB_DEV_REST_ENDPOINT


def dtype(*args, **kwargs):
    return np.dtype(*args, **kwargs)
