from deeplake.core.linked_tiled_sample import LinkedTiledSample
from typing import Optional, Dict
import numpy as np


def link_tiled(
    path_array: np.ndarray,
    creds_key: Optional[str] = None,
) -> LinkedTiledSample:
    """Utility that stores links to multiple images that together form a big tile. These images must all have the exact same dimension. Used to add data to a Deep Lake Dataset without copying it. See :ref:`Link htype`.

    Supported file types::

        Image: "bmp", "dib", "gif", "ico", "jpeg", "jpeg2000", "pcx", "png", "ppm", "sgi", "tga", "tiff", "webp", "wmf", "xbm"

    Args:
        path_array (np.ndarray): N dimensional array of paths to the data, with paths corresponding to respective tiles.
        creds_key (optional, str): The credential key to use to read data for this sample. The actual credentials are fetched from the dataset.

    Returns:
        LinkedTiledSample: LinkedTiledSample object that stores path_array and creds.

    """
    return LinkedTiledSample(path_array, creds_key)
