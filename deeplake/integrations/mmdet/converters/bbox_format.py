import numpy as np


def get_bbox_format(bbox, bbox_info):
    bbox_info = bbox_info.get("coords", {})
    mode = bbox_info.get("mode", "LTWH")
    type = bbox_info.get("type", "pixel")

    if len(bbox_info) == 0 and np.mean(bbox) < 1:
        mode = "CCWH"
        type = "fractional"
    return (mode, type)


def first_non_empty(bboxes):
    for box in bboxes:
        if len(box):
            return box
    raise ValueError("Empty bboxes")