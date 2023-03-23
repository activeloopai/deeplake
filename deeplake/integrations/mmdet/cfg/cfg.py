from typing import Any, Dict

import deeplake
from .cfg_object import CfgObject


def load(cfg: Dict[str, Any]) -> deeplake.core.dataset.Dataset:
    """
    Loads a deeplake dataset from DeepLake based on the given configuration.

    Args:
        cfg (Dict[str, Any]): The configuration dictionary.

    Returns:
        deeplake.core.dataset.Dataset: The loaded dataset.
    """
    cfg_obj = CfgObject(cfg)
    return cfg_obj.load()
