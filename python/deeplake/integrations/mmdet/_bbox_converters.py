import numpy as np


def yolo_pixel_2_pascal_pixel(boxes, shape):
    """Convert YOLO center/size boxes in pixels to Pascal VOC corner boxes."""
    pascal_boxes = np.empty((0, 4), dtype=boxes.dtype)
    if boxes.size != 0:
        x_top = boxes[:, 0] - boxes[:, 2] / 2
        y_top = boxes[:, 1] - boxes[:, 3] / 2
        x_bottom = boxes[:, 0] + boxes[:, 2] / 2
        y_bottom = boxes[:, 1] + boxes[:, 3] / 2
        pascal_boxes = np.stack((x_top, y_top, x_bottom, y_bottom), axis=1)
    return pascal_boxes
