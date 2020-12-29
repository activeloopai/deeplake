import numpy as np


from .collections import dataset
from .collections.dataset.core import Transform
from .collections import tensor
from .collections.dataset import load as load_v0
from .collections.client_manager import init
import hub.config
from hub.api.dataset import Dataset
from hub.compute import transform
from hub.log import logger
import traceback
from hub.exceptions import DaskModuleNotInstalledException, HubDatasetNotFoundException


def local_mode():
    hub.config.HUB_REST_ENDPOINT = hub.config.HUB_LOCAL_REST_ENDPOINT


def dev_mode():
    hub.config.HUB_REST_ENDPOINT = hub.config.HUB_DEV_REST_ENDPOINT


def dtype(*args, **kwargs):
    return np.dtype(*args, **kwargs)


def load(tag):
    """
    Load a dataset from repository using given tag

    Args:
        tag: string
        using {username}/{dataset} format or file system, s3://, gcs://

    Notes
    ------
    It will try to load using old version and fall off on newer version

    """
    try:
        ds = load_v0(tag)
        logger.warning(
            "Deprecated Warning: Given dataset is using deprecated format v0.x. Please convert to v1.x version upon availability."
        )
        return ds
    except ImportError:
        raise DaskModuleNotInstalledException
    except HubDatasetNotFoundException:
        raise
    except Exception as e:
        pass
        # logger.warning(traceback.format_exc() + str(e))

    return Dataset(tag)
