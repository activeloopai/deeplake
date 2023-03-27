import deeplake.integrations.mmdet.converters.coco_format_converters as coco_format_converters
import deeplake.integrations.mmdet.converters.bbox_format as bbox_format

from typing import Any, Callable, Dict, Tuple, List

BBOX_FORMAT_TO_COCO_CONVERTER: Dict[Tuple[str, str], Callable] = {
    ("LTWH", "pixel"): lambda x, y: x,
    ("LTWH", "fractional"): coco_format_converters.coco_frac_2_coco_pixel,
    ("LTRB", "pixel"): coco_format_converters.pascal_pixel_2_coco_pixel,
    ("LTRB", "fractional"): coco_format_converters.pascal_frac_2_coco_pixel,
    ("CCWH", "pixel"): coco_format_converters.yolo_pixel_2_coco_pixel,
    ("CCWH", "fractional"): coco_format_converters.yolo_frac_2_coco_pixel,
}


def convert(
    bbox: List[float], bbox_info: Dict[str, Dict[str, str]], shape: Tuple[int, int]
) -> List[float]:
    """
    Convert bounding boxes to COCO format (top-left x, top-left y, width, height)
    in pixel coordinates using the appropriate converter function based on the provided bbox_format.

    Args:
        bbox (List[float]): Bounding boxes to be converted.
        bbox_format (Tuple[str, str]): The format of the input bounding boxes.
            First element of the tuple represents the mode (e.g. 'LTWH', 'LTRB', 'CCWH').
            Second element of the tuple represents the type (e.g. 'pixel', 'fractional').
        images (Any): Images associated with the bounding boxes.

    Returns:
        List[float]: Bounding boxes in COCO format.
    """
    bbox_info = bbox_format.get_bbox_format(bbox, bbox_info)
    converter = BBOX_FORMAT_TO_COCO_CONVERTER[bbox_info]
    return converter(bbox, shape)
