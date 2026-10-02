import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).parents[1]
    / "deeplake"
    / "integrations"
    / "mmdet"
    / "_bbox_converters.py"
)
SPEC = importlib.util.spec_from_file_location("mmdet_bbox_converters", MODULE_PATH)
CONVERTERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONVERTERS)


def test_yolo_pixel_conversion_preserves_half_pixels_for_odd_dimensions():
    boxes = np.array([[50, 25, 21, 11]], dtype=np.float32)

    result = CONVERTERS.yolo_pixel_2_pascal_pixel(boxes, (51, 101))

    np.testing.assert_allclose(result, [[39.5, 19.5, 60.5, 30.5]])


def test_yolo_pixel_conversion_does_not_floor_sub_two_pixel_boxes():
    boxes = np.array([[10, 8, 1, 1]], dtype=np.float32)

    result = CONVERTERS.yolo_pixel_2_pascal_pixel(boxes, (16, 20))

    np.testing.assert_allclose(result, [[9.5, 7.5, 10.5, 8.5]])
