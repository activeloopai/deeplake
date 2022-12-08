import numpy as np
from typing import Dict, Any, Optional

from deeplake.core.tensor import Tensor


def coco_to_deeplake(
    coco_key: str,
    value: Any,
    destination_tensor: Tensor,
    category_lookup: Optional[Dict] = None,
):
    """Takes a key-value pair from coco data and converts it to data in Deep Lake compatible format"""
    dtype = destination_tensor.meta.dtype

    if coco_key == "bbox":
        assert len(value) == 4
        return np.array(value, dtype=dtype)
    elif coco_key == "segmentation":
        try:
            # Currently having only ONE polygon is supported.
            # Multiple polygons are under same label, but there is only a single bbox.
            # Can not think of a way to support multiple polygons for same label on the same image, other than converting to a mask.
            return np.array(value[0], dtype=dtype).reshape(
                (len(value[0]) // 2), 2
            )  # Convert to array of x-y coordinates
        except KeyError:
            """KeyError happens if the value is NOT a list of polygons.
            Returning None does not work, as polygon is erroring out during shape validation.
            Returning an empty array works, but the data is wrong... Alternatively, whole sample may be skipped.
            """

            return np.array([[0, 0]], dtype=dtype)

    elif coco_key == "category_id":
        if category_lookup is None:
            return value
        else:
            return category_lookup[str(value)]

    elif coco_key == "keypoints":
        return np.array(value, dtype=dtype)

    return value
