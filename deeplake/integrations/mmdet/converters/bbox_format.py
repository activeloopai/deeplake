import numpy as np
from typing import Any, Dict, List, Tuple, Union


def get_bbox_format(
    bbox: List[float], bbox_info: Dict[str, Dict[str, str]]
) -> Tuple[str, str]:
    """
    Determines the bounding box format based on the given bounding box and information.

    Args:
        bbox (List[float]): The bounding box coordinates.
        bbox_info (Dict[str, Dict[str, str]]): The bounding box information dictionary.

    Returns:
        Tuple[str, str]: A tuple containing the mode (e.g. CCWH, TLRB, ...) and type (e.g. pixel, fractional) of the bounding box format.
    """
    bbox_info = bbox_info.get("coords", {})
    mode = bbox_info.get("mode", "LTWH")
    type = bbox_info.get("type", "pixel")

    if len(bbox_info) == 0 and np.mean(bbox) < 1:
        mode = "CCWH"
        type = "fractional"
    return (mode, type)


def first_non_empty(bboxes: List[List[Union[int, float]]]) -> List[Union[int, float]]:
    """
    Finds and returns the first non-empty bounding box in a list of bounding boxes.

    Args:
        bboxes (List[List[Union[int, float]]]): A list of bounding boxes.

    Returns:
        List[Union[int, float]]: The first non-empty bounding box found.

    Raises:
        ValueError: If all bounding boxes are empty.
    """
    for box in bboxes:
        if len(box):
            return box
    raise ValueError("Empty bboxes")
