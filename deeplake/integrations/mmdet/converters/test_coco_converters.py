import pytest
from typing import List, Tuple

import numpy as np

import deeplake.integrations.mmdet.converters.coco_format as coco_format
import deeplake.integrations.mmdet.converters.bbox_format as bbox_format


BBOX_INFO_TO_CORRECT_BBOX = {
    ("LTWH", "pixel"): np.array([[4, 5, 2, 2]]),
    ("LTWH", "fractional"): np.array([[10, 12.5, 5, 5]]),
    ("LTRB", "pixel"):  np.array([[4, 5, 2, 2]]),
    ("LTRB", "fractional"):  np.array([[8, 10, 4, 4]]),
    ("CCWH", "pixel"): np.array([[3, 4, 2, 2]]),
    ("CCWH", "fractional"): np.array([[6, 8, 4, 4]]),
}

@pytest.mark.parametrize(
    "bbox, bbox_info, images",
    [
        [np.array([[[4, 5, 6, 7]]]), {"coords": {"mode": "LTRB", "type": "pixel"}}, [np.zeros((10, 10))]],
        [np.array([[[0.4, 0.5, 0.6, 0.7]]]), {"coords": {"mode": "LTRB", "type": "fractional"}}, [np.zeros((20, 20))]],
        [np.array([[[4, 5, 2, 2]]]), {"coords": {"mode": "CCWH", "type": "pixel"}}, [np.zeros((10, 10))]],
        [np.array([[[0.4, 0.5, 0.2, 0.2]]]), {"coords": {"mode": "CCWH", "type": "fractional"}}, [np.zeros((20, 20))]],
        [np.array([[[4, 5, 2, 2]]]), {"coords": {"mode": "LTWH", "type": "pixel"}}, [np.zeros((10, 10))]],
        [np.array([[[0.4, 0.5, 0.2, 0.2]]]), {"coords": {"mode": "LTWH", "type": "fractional"}}, [np.zeros((25, 25))]],
    ],
)
def test_conveter(bbox: np.ndarray, bbox_info: Tuple, images: List[np.ndarray]):
    converted_bbox = coco_format.convert(bbox, bbox_info, images)

    bbox_info_tuple = bbox_format.get_bbox_format(bbox, bbox_info)
    targ_bbox = BBOX_INFO_TO_CORRECT_BBOX[bbox_info_tuple]

    np.testing.assert_array_equal(converted_bbox[0], targ_bbox)
