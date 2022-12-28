import numpy as np
from typing import Dict, Any, Optional

from deeplake.core.tensor import Tensor
from deeplake.util.exceptions import IngestionError
from deeplake.client.log import logger


def coco_to_deeplake(
    coco_key: str,
    value: Any,
    destination_tensor: Tensor,
    category_lookup: Optional[Dict] = None,
):
    """Takes a key-value pair from coco data and converts it to data in Deep Lake compatible format"""
    dtype = destination_tensor.meta.dtype

    if coco_key == "bbox":
        if len(value) != 4:
            raise IngestionError(
                f"Invalid bbox encountered in key {coco_key}. Bbox must have 4 values."
            )

        return np.array(value, dtype=dtype)
    elif coco_key == "segmentation":
        if not isinstance(value, list):
            raise IngestionError(
                f"Invalid value encountered in key {coco_key}. Segmentation must be a list of polygons."
            )

        # Currently having only ONE polygon is supported.
        # Multiple polygons are under same label, but there is only a single bbox.
        if len(value) > 1:
            logger.warning(
                f"Multiple polygons are not supported in key {coco_key}. Only the first one will be used."
            )

        if len(value) == 0:
            return None

        return np.array(value[0], dtype=dtype).reshape(
            (len(value[0]) // 2), 2
        )  # Convert to array of x-y coordinates

    elif coco_key == "category_id":
        if category_lookup is None:
            return value
        else:
            return category_lookup[str(value)]

    elif coco_key == "keypoints":
        return np.array(value, dtype=dtype)

    return value
