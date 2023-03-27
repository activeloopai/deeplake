import numpy as np

import deeplake.integrations.mmdet.converters.bbox_format as bbox_format
import deeplake.integrations.mmdet.converters.pascal_format_converters as pascal_format_converters


BBOX_FORMAT_TO_PASCAL_CONVERTER = {
    ("LTWH", "pixel"): pascal_format_converters.coco_pixel_2_pascal_pixel,
    ("LTWH", "fractional"): pascal_format_converters.coco_frac_2_pascal_pixel,
    ("LTRB", "pixel"): lambda x, y: x,
    ("LTRB", "fractional"): pascal_format_converters.pascal_frac_2_pascal_pixel,
    ("CCWH", "pixel"): pascal_format_converters.yolo_pixel_2_pascal_pixel,
    ("CCWH", "fractional"): pascal_format_converters.yolo_frac_2_pascal_pixel,
}


def convert(bbox: np.ndarray, bbox_info: dict, shape: tuple) -> np.ndarray:
    """
    Converts bounding box coordinates to Pascal format.

    Args:
        bbox (np.ndarray): A Numpy array containing bounding box coordinates.
        bbox_info (dict): A dictionary containing information about the bounding box format. Ex: {"mode": "LTWH", "type": "pixel"}
        shape (tuple): A tuple containing the shape of the image.

    Returns:
        np.ndarray: A Numpy array containing bounding box coordinates in Pascal format.
    """
    bbox_info = bbox_format.get_bbox_format(bbox, bbox_info)
    converter = BBOX_FORMAT_TO_PASCAL_CONVERTER[bbox_info]
    return converter(bbox, shape)
