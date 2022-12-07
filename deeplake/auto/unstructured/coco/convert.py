import numpy as np
from typing import Dict, Any, Optional

from deeplake.core.tensor import Tensor


def coco_2_deeplake(
    coco_key: str,
    value: Any,
    destination_tensor: Tensor,
    category_lookup: Optional[Dict] = None,
):
    """Takes a key-value pair from coco data and converts it to data in Deep Lake format
    as per the key types in coco and array shape rules in Deep Lake"""
    dtype = destination_tensor.meta.dtype

    if isinstance(value, list) and len(value) == 0:
        raise Exception("Empty value for key: " + coco_key)

    if coco_key == "bbox":
        assert len(value) == 4
        return np.array(value, dtype=dtype)
    elif coco_key == "segmentation":
        # Make sure there aren't multiple segementations per single value, because multiple things will break
        # # if len(value) > 1:
        #     print("MULTIPLE SEGMENTATIONS PER OBJECT")

        try:
            return np.array(value[0], dtype=dtype).reshape((len(value[0]) // 2), 2)
        except KeyError:
            return np.array([[0, 0]], dtype=dtype)

    elif coco_key == "category_id":
        if category_lookup is None:
            return value
        else:
            return category_lookup[str(value)]

    elif coco_key == "keypoints":
        return np.array(value, dtype=dtype)

    return value
